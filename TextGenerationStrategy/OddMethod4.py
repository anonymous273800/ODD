# #### **(5) D Controls Both (Unified Penalty)**
#
# * Combine D and C into one term:
#   ( c'*{\text{LM}} = c*{\text{LM}} (1 - (0.7D + 0.3(1-C))^2) )
# * No direct reward on Trie.
# * Purpose: unify both disagreement and continuity into a *single adaptive penalty* for the LLM confidence.
from tqdm import tqdm
import math
import Utils.Constants
from DataFineTuning import AplacaFineTuner
from Utils import Util
import torch
import torch.nn.functional as F
import numpy as np
torch.set_printoptions(threshold=torch.inf)

def compute_continuity_C(last_matched_count, max_memory=3):
    C = 1 - np.exp(-last_matched_count/max_memory)
    return float(C)


def compute_llm_confidence(p_lm_b, eps=1e-12):
    p = p_lm_b.clamp_min(eps)
    H = -(p * p.log()).sum()
    H_max = math.log(float(p.shape[0]))
    c_lm = 1.0 - (H / (H_max + eps))
    return float(max(0.0, min(1.0, c_lm)))



def compute_adaptive_temperature_exact(logits, top_trie_prob,
                                       tol=1e-4, max_iter=30):
    """
    Adaptively compute temperature T so that the top probability of
    softmax(logits/T) matches the top probability from the trie distribution.
    """
    V = logits.numel()
    eps = 1e-12
    uniform = 1.0 / V

    if top_trie_prob is None or top_trie_prob <= 0.0:
        return 1.0  # fallback if trie has no signal

    # clip the target into feasible range
    target = min(1.0 - eps, max(uniform + eps, top_trie_prob))

    def top_prob_at(T):
        p = torch.softmax(logits / T, dim=-1)
        return float(p.max())

    # bracket
    T_lo, T_hi = 0.05, 50.0
    p_lo = top_prob_at(T_lo)  # very sharp
    p_hi = top_prob_at(T_hi)  # very flat

    if target > p_lo:
        return T_lo
    if target < p_hi:
        return T_hi

    for _ in range(max_iter):
        T_mid = 0.5 * (T_lo + T_hi)
        p_mid = top_prob_at(T_mid)
        if abs(p_mid - target) <= tol:
            return T_mid
        if p_mid > target:
            T_lo = T_mid
        else:
            T_hi = T_mid
    return 0.5 * (T_lo + T_hi)

# ---------- (1) Compute disagreement between LLM and Trie ----------
def compute_disagreement_D(p_lm, p_trie, k=8, conf_lm=None):
    """
    Compute disagreement D_t between LLM and Trie distributions
    using Jensen–Shannon divergence over their top-k tokens,
    with optional confidence-based clipping.
    """
    if torch.all(p_trie == 0) or p_trie.sum() == 0:
        return 0.0

    # effective top-k
    nonzero_trie = (p_trie > 0).sum().item()
    k_eff = max(1, min(k, nonzero_trie))

    # top-k union
    topk_lm = torch.topk(p_lm, k_eff).indices.tolist()
    topk_trie = torch.topk(p_trie, k_eff).indices.tolist()
    all_indices = list(set(topk_lm + topk_trie))

    p_lm_k = p_lm[all_indices]
    p_trie_k = p_trie[all_indices]
    p_lm_k /= (p_lm_k.sum() + 1e-12)
    p_trie_k /= (p_trie_k.sum() + 1e-12)

    p = p_lm_k.detach().cpu().numpy() + 1e-12
    q = p_trie_k.detach().cpu().numpy() + 1e-12

    if np.allclose(p, q) or np.std(p) < 1e-8 or np.std(q) < 1e-8:
        return 0.0

    # Jensen–Shannon divergence
    m = 0.5 * (p + q)
    jsd = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    D = float(np.sqrt(jsd))
    D = min(1.0, D)

    # --- confidence-based guard ---
    if conf_lm is not None:
        top2 = torch.topk(p_lm, 2).values
        margin = float(top2[0] - top2[1])
        if margin > 0.15:  # high certainty → cap D
            D = min(D, 0.4)

    return D

def generate_greedy4(
        model,
        tokenizer,
        idx,
        max_new_tokens,
        context_size,
        trie,
        scoring_config=None,
        eos_id=None
):
    scoring_args = {k: scoring_config[k] for k in ["w_len", "w_freq", "w_recency"] if k in scoring_config}
    last_matched_count = [0 for _ in range(idx.size(0))]
    is_first_misalignment=False
    for i in range(max_new_tokens):
        end_flag = False
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # (B, V)
        batch_size, vocab_size = logits.shape
        p_lm = torch.zeros_like(logits)


        for b in range(batch_size):
            prefix_tokens = idx[b].tolist()
            ranked = trie.score_next_tokens(prefix_tokens, **scoring_args)

            if not ranked:
                p_lm[b] = F.softmax(logits[b], dim=-1)
                continue

            top_trie_token, top_trie_score, _ = ranked[0]

            # --- Build trie probability distribution ---
            trie_scores = torch.zeros(vocab_size, device=logits.device)
            for tok, score, _ in ranked:
                trie_scores[tok] = max(score, 0.0)
            p_trie = Util.normalize_trie_scores(trie_scores) if trie_scores.max() > 0 else torch.zeros_like(trie_scores)


            # Compute LLM Confidence correctly (using probabilities)
            p_lm_base = F.softmax(logits[b], dim=-1)
            c_lm_base = compute_llm_confidence(p_lm_base)

            # --- Align LLM temperature ---
            T = compute_adaptive_temperature_exact(logits[b], top_trie_prob=top_trie_score)
            p_lm_temp = F.softmax(logits[b] / T, dim=-1)
            print(f"Adaptive T (exact): {T:.4f}, target trie max={top_trie_score:.4f}, llm_max={p_lm_temp.max().item():.4f}")

            # --- Compute base confidences ---

            c_trie_base = float(p_trie.max().item()) if p_trie.sum() > 0 else 0.0
            print("LLM Confidence", c_lm_base)
            print("Trie Confidence", c_trie_base)

            # --- Compute disagreement D_t and continuity C_t ---
            D = compute_disagreement_D(p_lm_temp, p_trie, k=8)
            C = compute_continuity_C(last_matched_count[b])


            # --- Update last_matched_count ---
            top_llm_token = torch.argmax(p_lm_temp).item()
            top_llm_prob = p_lm_temp[top_llm_token].item()

            # --- Adjust confidences ---
            # c_lm_adj = c_lm_base * (1 - D**2)
            # c_trie_adj = c_trie_base + (1 - c_trie_base) * 0.25 * C

            c_lm_adj = max(0.0, min(1.0, c_lm_base * (1 - (.7*D + .3*(1 - C))**2)))

            print("LLM Conficence (After Panelizing)", c_lm_adj)
            # print("Trie Confidence (After Reqarding)", c_trie_adj)


            # --- Compute gamma with adjusted confidences ---
            gamma = c_lm_adj / (c_lm_adj + c_trie_base + 1e-12)

            # --- First misalignment override ---
            if not is_first_misalignment and top_llm_token != top_trie_token:
                gamma = 1.0  # fully prefer LLM
                is_first_misalignment = True
                print(f"[FIRST MISMATCH @ step {i}] → Forcing LLM dominance (gamma=1.0)")

            # --- Mix ---
            p_mix = gamma * p_lm_temp + (1 - gamma) * p_trie
            p_mix = p_mix / (p_mix.sum() + 1e-12)

            top_mix_token = torch.argmax(p_mix).item()
            top_mix_prob = p_mix[top_mix_token].item()

            if top_mix_token == top_trie_token:
                last_matched_count[b] += 1
            else:
                last_matched_count[b] = 0

            if top_mix_token == eos_id: end_flag = True

            print(f"c_lm_adj: {c_lm_adj:.3f} | gamma: {gamma:.3f} | D={D:.3f}, C={C:.3f}")
            print("SUMMARY LLM:", top_llm_token, Util.token_ids_to_text(torch.tensor([[top_llm_token]], device=logits.device), tokenizer), "(", top_llm_prob, ")")
            print("SUMMARY TRIE:", top_trie_token, Util.token_ids_to_text(torch.tensor([[top_trie_token]], device=logits.device), tokenizer), "(", top_trie_score, ")")
            print("SUMMARY MIX:", top_mix_token, Util.token_ids_to_text(torch.tensor([[top_mix_token]], device=logits.device), tokenizer), "(", top_mix_prob, ")")

            print("----- ----- ---- --- ")

            p_lm[b] = p_mix.clone()

        idx_next = torch.argmax(p_lm, dim=-1, keepdim=True)

        if end_flag:
            idx_next = torch.full_like(idx_next, eos_id)

        if eos_id is not None and (idx_next.squeeze(-1) == eos_id).all():
            break
        idx = torch.cat((idx, idx_next), dim=1)

    print()
    print("idx F:", idx, "Decoded F:", Util.token_ids_to_text(idx, tokenizer))
    print("############################################################")
    return idx



# #### **(5) D Controls Both (Unified Penalty)**
#
# * Combine D and C into one term:
#   ( c'*{\text{LM}} = c*{\text{LM}} (1 - (0.7D + 0.3(1-C))^2) )
# * No direct reward on Trie.
# * Purpose: unify both disagreement and continuity into a *single adaptive penalty* for the LLM confidence.
from tqdm import tqdm
import math
import Utils.Constants
from DataFineTuning import AplacaFineTuner
from Utils import Util
import torch
import torch.nn.functional as F
import numpy as np
torch.set_printoptions(threshold=torch.inf)

def compute_continuity_C(last_matched_count, max_memory=3):
    C = 1 - np.exp(-last_matched_count/max_memory)
    return float(C)


def compute_llm_confidence(p_lm_b, eps=1e-12):
    p = p_lm_b.clamp_min(eps)
    H = -(p * p.log()).sum()
    H_max = math.log(float(p.shape[0]))
    c_lm = 1.0 - (H / (H_max + eps))
    return float(max(0.0, min(1.0, c_lm)))



def compute_adaptive_temperature_exact(logits, top_trie_prob,
                                       tol=1e-4, max_iter=30):
    """
    Adaptively compute temperature T so that the top probability of
    softmax(logits/T) matches the top probability from the trie distribution.
    """
    V = logits.numel()
    eps = 1e-12
    uniform = 1.0 / V

    if top_trie_prob is None or top_trie_prob <= 0.0:
        return 1.0  # fallback if trie has no signal

    # clip the target into feasible range
    target = min(1.0 - eps, max(uniform + eps, top_trie_prob))

    def top_prob_at(T):
        p = torch.softmax(logits / T, dim=-1)
        return float(p.max())

    # bracket
    T_lo, T_hi = 0.05, 50.0
    p_lo = top_prob_at(T_lo)  # very sharp
    p_hi = top_prob_at(T_hi)  # very flat

    if target > p_lo:
        return T_lo
    if target < p_hi:
        return T_hi

    for _ in range(max_iter):
        T_mid = 0.5 * (T_lo + T_hi)
        p_mid = top_prob_at(T_mid)
        if abs(p_mid - target) <= tol:
            return T_mid
        if p_mid > target:
            T_lo = T_mid
        else:
            T_hi = T_mid
    return 0.5 * (T_lo + T_hi)

# ---------- (1) Compute disagreement between LLM and Trie ----------
def compute_disagreement_D(p_lm, p_trie, k=8, conf_lm=None):
    """
    Compute disagreement D_t between LLM and Trie distributions
    using Jensen–Shannon divergence over their top-k tokens,
    with optional confidence-based clipping.
    """
    if torch.all(p_trie == 0) or p_trie.sum() == 0:
        return 0.0

    # effective top-k
    nonzero_trie = (p_trie > 0).sum().item()
    k_eff = max(1, min(k, nonzero_trie))

    # top-k union
    topk_lm = torch.topk(p_lm, k_eff).indices.tolist()
    topk_trie = torch.topk(p_trie, k_eff).indices.tolist()
    all_indices = list(set(topk_lm + topk_trie))

    p_lm_k = p_lm[all_indices]
    p_trie_k = p_trie[all_indices]
    p_lm_k /= (p_lm_k.sum() + 1e-12)
    p_trie_k /= (p_trie_k.sum() + 1e-12)

    p = p_lm_k.detach().cpu().numpy() + 1e-12
    q = p_trie_k.detach().cpu().numpy() + 1e-12

    if np.allclose(p, q) or np.std(p) < 1e-8 or np.std(q) < 1e-8:
        return 0.0

    # Jensen–Shannon divergence
    m = 0.5 * (p + q)
    jsd = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    D = float(np.sqrt(jsd))
    D = min(1.0, D)

    # --- confidence-based guard ---
    if conf_lm is not None:
        top2 = torch.topk(p_lm, 2).values
        margin = float(top2[0] - top2[1])
        if margin > 0.15:  # high certainty → cap D
            D = min(D, 0.4)

    return D

def generate_greedy4(
        model,
        tokenizer,
        idx,
        max_new_tokens,
        context_size,
        trie,
        scoring_config=None,
        eos_id=None
):
    scoring_args = {k: scoring_config[k] for k in ["w_len", "w_freq", "w_recency"] if k in scoring_config}
    last_matched_count = [0 for _ in range(idx.size(0))]

    for _ in range(max_new_tokens):
        end_flag = False
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # (B, V)
        batch_size, vocab_size = logits.shape
        p_lm = torch.zeros_like(logits)


        for b in range(batch_size):
            prefix_tokens = idx[b].tolist()
            ranked = trie.score_next_tokens(prefix_tokens, **scoring_args)

            if not ranked:
                p_lm[b] = F.softmax(logits[b], dim=-1)
                continue

            top_trie_token, top_trie_score, _ = ranked[0]

            # --- Build trie probability distribution ---
            trie_scores = torch.zeros(vocab_size, device=logits.device)
            for tok, score, _ in ranked:
                trie_scores[tok] = max(score, 0.0)
            p_trie = Util.normalize_trie_scores(trie_scores) if trie_scores.max() > 0 else torch.zeros_like(trie_scores)


            # Compute LLM Confidence correctly (using probabilities)
            p_lm_base = F.softmax(logits[b], dim=-1)
            c_lm_base = compute_llm_confidence(p_lm_base)

            # --- Align LLM temperature ---
            T = compute_adaptive_temperature_exact(logits[b], top_trie_prob=top_trie_score)
            p_lm_temp = F.softmax(logits[b] / T, dim=-1)
            print(f"Adaptive T (exact): {T:.4f}, target trie max={top_trie_score:.4f}, llm_max={p_lm_temp.max().item():.4f}")

            # --- Compute base confidences ---

            c_trie_base = float(p_trie.max().item()) if p_trie.sum() > 0 else 0.0
            print("LLM Confidence", c_lm_base)
            print("Trie Confidence", c_trie_base)

            # --- Compute disagreement D_t and continuity C_t ---
            D = compute_disagreement_D(p_lm_temp, p_trie, k=8)
            C = compute_continuity_C(last_matched_count[b])


            # --- Update last_matched_count ---
            top_llm_token = torch.argmax(p_lm_temp).item()
            top_llm_prob = p_lm_temp[top_llm_token].item()

            # --- Adjust confidences ---
            # c_lm_adj = c_lm_base * (1 - D**2)
            # c_trie_adj = c_trie_base + (1 - c_trie_base) * 0.25 * C

            c_lm_adj = max(0.0, min(1.0, c_lm_base * (1 - (.7*D + .3*(1 - C))**2)))

            print("LLM Conficence (After Panelizing)", c_lm_adj)
            # print("Trie Confidence (After Reqarding)", c_trie_adj)


            # --- Compute gamma with adjusted confidences ---
            gamma = c_lm_adj / (c_lm_adj + c_trie_base + 1e-12)

            # --- Mix ---
            p_mix = gamma * p_lm_temp + (1 - gamma) * p_trie
            p_mix = p_mix / (p_mix.sum() + 1e-12)

            top_mix_token = torch.argmax(p_mix).item()
            top_mix_prob = p_mix[top_mix_token].item()

            if top_mix_token == top_trie_token:
                last_matched_count[b] += 1
            else:
                last_matched_count[b] = 0

            if top_mix_token == eos_id: end_flag = True

            print(f"c_lm_adj: {c_lm_adj:.3f} | gamma: {gamma:.3f} | D={D:.3f}, C={C:.3f}")
            print("SUMMARY LLM:", top_llm_token, Util.token_ids_to_text(torch.tensor([[top_llm_token]], device=logits.device), tokenizer), "(", top_llm_prob, ")")
            print("SUMMARY TRIE:", top_trie_token, Util.token_ids_to_text(torch.tensor([[top_trie_token]], device=logits.device), tokenizer), "(", top_trie_score, ")")
            print("SUMMARY MIX:", top_mix_token, Util.token_ids_to_text(torch.tensor([[top_mix_token]], device=logits.device), tokenizer), "(", top_mix_prob, ")")

            print("----- ----- ---- --- ")

            p_lm[b] = p_mix.clone()

        idx_next = torch.argmax(p_lm, dim=-1, keepdim=True)

        if end_flag:
            idx_next = torch.full_like(idx_next, eos_id)

        if eos_id is not None and (idx_next.squeeze(-1) == eos_id).all():
            break
        idx = torch.cat((idx, idx_next), dim=1)

    print()
    print("idx F:", idx, "Decoded F:", Util.token_ids_to_text(idx, tokenizer))
    print("############################################################")
    return idx

def response_strategy_based_extractor_and_saver_abrupt4(model, tokenizer, test_data, device, BASE_CONFIG, trie1, trie2, SCORING_CONFIG):
    n_test = len(test_data)
    half = n_test // 2

    test_data = list(test_data)  # Convert HF dataset to a list of dicts
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        input_text = AplacaFineTuner.format_input(entry)
        input_ids = Util.text_to_token_ids(input_text, tokenizer).to(device)
        prompt_len = input_ids.size(1)

        if i < half:
            trie = trie1
            print("TRIE is TRIE1 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        else:
            trie = trie2
            print("TRIE is TRIE2 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


        print(".....................")
        print("input_text: ", input_text)
        print(".....................")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ODD_4 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        token_ids = generate_greedy4(
            model=model,
            tokenizer=tokenizer,
            idx=input_ids,
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            trie=trie,
            scoring_config=SCORING_CONFIG,
            eos_id=50256
        )

        # Optional check:
        print("Prompt length:", prompt_len, " Generated length:", token_ids.size(1), " Continuation length:", token_ids.size(1) - prompt_len)

        # Decode only continuation
        continuation_tokenstoken_ids = token_ids[:, prompt_len:]
        response_text = Util.token_ids_to_text(continuation_tokenstoken_ids, tokenizer)


        # Clean up special markers
        response_text = response_text.replace("### Response:", "").replace("<|endoftext|>", "").strip()
        print("model response=>", response_text)

        test_data[i]["model_response"] = response_text
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ END ODD_4 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")



        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("response=>", test_data[i]["response"])
    Util.store_in_json_file(test_data, Utils.Constants.RESPONSE_SPINE_FILE_NAME4)




def response_strategy_based_extractor_and_saver_abrupt_all_4(model, tokenizer, test_data, device, BASE_CONFIG, trie1, trie2, SCORING_CONFIG):
    n_test = len(test_data)
    half = n_test // 2

    test_data = list(test_data)  # Convert HF dataset to a list of dicts
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        input_text = AplacaFineTuner.format_input(entry)
        input_ids = Util.text_to_token_ids(input_text, tokenizer).to(device)
        prompt_len = input_ids.size(1)

        if i < half:
            trie = trie1
            print("TRIE is TRIE1 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        else:
            trie = trie2
            print("TRIE is TRIE2 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


        print(".....................")
        print("input_text: ", input_text)
        print(".....................")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ODD_4 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        token_ids = generate_greedy4(
            model=model,
            tokenizer=tokenizer,
            idx=input_ids,
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            trie=trie,
            scoring_config=SCORING_CONFIG,
            eos_id=50256
        )

        # Optional check:
        print("Prompt length:", prompt_len, " Generated length:", token_ids.size(1), " Continuation length:", token_ids.size(1) - prompt_len)

        # Decode only continuation
        continuation_tokenstoken_ids = token_ids[:, prompt_len:]
        response_text = Util.token_ids_to_text(continuation_tokenstoken_ids, tokenizer)


        # Clean up special markers
        response_text = response_text.replace("### Response:", "").replace("<|endoftext|>", "").strip()
        print("model response=>", response_text)

        test_data[i]["model_response"] = response_text
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ END ODD_4 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")



        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("response=>", test_data[i]["response"])
    Util.store_in_json_file(test_data, Utils.Constants.RESPONSE_SPINE_FILE_NAME_ALL_4)


# === Unified Penalty (Variant #5) Main Test ===
def test_case(p_lm, p_trie, last_matched_count):
    c_lm = compute_llm_confidence(p_lm)
    c_trie = float(p_trie.max().item())
    D = compute_disagreement_D(p_lm, p_trie, k=8, conf_lm=c_lm)
    C = compute_continuity_C(last_matched_count)

    # Unified penalty on LLM confidence
    c_lm_adj = max(0.0, min(1.0, c_lm * (1 - (0.7 * D + 0.3 * (1 - C))**2)))

    gamma = c_lm_adj / (c_lm_adj + c_trie + 1e-12)
    top_lm = torch.argmax(p_lm).item()
    top_trie = torch.argmax(p_trie).item()
    winner = "LLM" if gamma > 0.5 else "Trie"

    print(f"\n=== Test Case ===")
    print(f"p_lm={p_lm.tolist()} | p_trie={p_trie.tolist()} | matches={last_matched_count}")
    print(f"c_lm={c_lm:.3f}, c_trie={c_trie:.3f}, D={D:.3f}, C={C:.3f}")
    print(f"c_lm_adj={c_lm_adj:.3f}, gamma={gamma:.3f}")
    print(f"Top LLM token={top_lm}, Top Trie token={top_trie} → Winner: {winner}")


if __name__ == "__main__":
    print("\n--- Variant #5: Unified Penalty (D + C) Tests ---")

    # 1️⃣ Highly confident LLM (low D, low C → LLM should win)
    test_case(torch.tensor([0.97, 0.02, 0.01]),
              torch.tensor([0.4, 0.3, 0.3]),
              last_matched_count=0)

    # 2️⃣ Strong Trie (high C continuity → Trie wins)
    test_case(torch.tensor([0.34, 0.33, 0.33]),
              torch.tensor([0.85, 0.10, 0.05]),
              last_matched_count=3)

    # 3️⃣ Conflicting but moderate disagreement (balanced)
    test_case(torch.tensor([0.7, 0.2, 0.1]),
              torch.tensor([0.2, 0.7, 0.1]),
              last_matched_count=1)

    # 4️⃣ Progressive continuity effect (observe C’s influence)
    for m in [0, 2, 5]:
        test_case(torch.tensor([0.6, 0.25, 0.15]),
                  torch.tensor([0.7, 0.2, 0.1]),
                  last_matched_count=m)

    # 5️⃣ Confident but conflicting LLM (both peaked)
    test_case(torch.tensor([0.95, 0.03, 0.02]),
              torch.tensor([0.02, 0.95, 0.03]),
              last_matched_count=0)
