# Remove C at all
from tqdm import tqdm
import math
import Utils.Constants
from DataFineTuning import AplacaFineTuner
from Utils import Util
import torch
import torch.nn.functional as F
import numpy as np
import re
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

# # ---------- (1) Compute disagreement between LLM and Trie ----------
def compute_disagreement_D(p_lm, p_trie, k=5):
    """
    Compute disagreement D_t between LLM and Trie distributions
    using Jensen–Shannon divergence over their top-k tokens.
    Handles empty or single-token Trie gracefully.
    """
    # Check for trivial cases
    if torch.all(p_trie == 0) or p_trie.sum() == 0:
        return 0.0  # No Trie signal -> no disagreement

    # Determine adaptive k
    nonzero_trie = (p_trie > 0).sum().item()
    k_eff = max(1, min(k, nonzero_trie))  # effective top-k size

    # Get top-k token indices from both distributions
    topk_lm = torch.topk(p_lm, k_eff).indices.tolist()
    topk_trie = torch.topk(p_trie, k_eff).indices.tolist()
    all_indices = list(set(topk_lm + topk_trie))

    # Extract restricted distributions
    p_lm_k = p_lm[all_indices]
    p_trie_k = p_trie[all_indices]

    # Normalize locally (safe normalization)
    p_lm_k = p_lm_k / (p_lm_k.sum() + 1e-12)
    p_trie_k = p_trie_k / (p_trie_k.sum() + 1e-12)

    # Convert to numpy for divergence computation
    p = p_lm_k.detach().cpu().numpy() + 1e-12
    q = p_trie_k.detach().cpu().numpy() + 1e-12

    # Handle uniform or degenerate cases
    if np.allclose(p, q) or np.std(p) < 1e-8 or np.std(q) < 1e-8:
        return 0.0

    # Compute Jensen–Shannon divergence
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    jsd = 0.5 * (kl_pm + kl_qm)

    # Scale to [0,1]
    D = float(np.sqrt(jsd))
    return min(1.0, D)


# def compute_disagreement_D(p_lm_temp, p_trie, k=5):
#     topk_lm = torch.topk(p_lm_temp.view(-1), k).indices.tolist()
#     topk_trie = torch.topk(p_trie.view(-1), k).indices.tolist()
#     all_indices = list(set(topk_lm + topk_trie))
#     diff = torch.abs(p_lm_temp.view(-1)[all_indices] - p_trie.view(-1)[all_indices])
#     return float(diff.mean().item())


# def generate_greedy1(
#         model,
#         tokenizer,
#         idx,
#         max_new_tokens,
#         context_size,
#         trie,
#         scoring_config=None,
#         eos_id=None
# ):
#     scoring_args = {k: scoring_config[k] for k in ["w_len", "w_freq", "w_recency"] if k in scoring_config}
#     last_matched_count = [0 for _ in range(idx.size(0))]
#     is_first_misalignment = False
#     for i in range(max_new_tokens):
#         end_flag = False
#         idx_cond = idx[:, -context_size:]
#
#         with torch.no_grad():
#             logits = model(idx_cond)
#         logits = logits[:, -1, :]  # (B, V)
#         batch_size, vocab_size = logits.shape
#         p_lm = torch.zeros_like(logits)
#
#
#         for b in range(batch_size):
#             prefix_tokens = idx[b].tolist()
#             ranked = trie.score_next_tokens(prefix_tokens, **scoring_args)
#
#             if not ranked:
#                 p_lm[b] = F.softmax(logits[b], dim=-1)
#                 continue
#
#             top_trie_token, top_trie_score, _ = ranked[0]
#
#             # --- Build trie probability distribution ---
#             trie_scores = torch.zeros(vocab_size, device=logits.device)
#             for tok, score, _ in ranked:
#                 trie_scores[tok] = max(score, 0.0)
#             p_trie = Util.normalize_trie_scores(trie_scores) if trie_scores.max() > 0 else torch.zeros_like(trie_scores)
#
#
#             # Compute LLM Confidence correctly (using probabilities)
#             p_lm_base = F.softmax(logits[b], dim=-1)
#             c_lm_base = compute_llm_confidence(p_lm_base)
#
#             # --- Align LLM temperature ---
#             T = compute_adaptive_temperature_exact(logits[b], top_trie_prob=top_trie_score)
#             p_lm_temp = F.softmax(logits[b] / T, dim=-1)
#             print(f"Adaptive T (exact): {T:.4f}, target trie max={top_trie_score:.4f}, llm_max={p_lm_temp.max().item():.4f}")
#
#             # --- Compute base confidences ---
#
#             c_trie_base = float(p_trie.max().item()) if p_trie.sum() > 0 else 0.0
#             print("LLM Confidence", c_lm_base)
#             print("Trie Confidence", c_trie_base)
#
#             # --- Compute disagreement D_t and continuity C_t ---
#             D = compute_disagreement_D(p_lm_temp, p_trie, k=5)
#             # C = compute_continuity_C(last_matched_count[b])
#
#
#             # --- Update last_matched_count ---
#             top_llm_token = torch.argmax(p_lm_temp).item()
#             top_llm_prob = p_lm_temp[top_llm_token].item()
#
#             # --- Adjust confidences ---
#             c_lm_adj = c_lm_base * (1 - D**2)
#             # c_trie_adj = c_trie_base + (1 - c_trie_base) * (c_trie_base**2) * C
#
#             print("LLM Conficence (After Panelizing)", c_lm_adj)
#             print("Trie Confidence (BASE AS WELL)", c_trie_base)
#
#
#             # --- Compute gamma with adjusted confidences ---
#             gamma = c_lm_adj / (c_lm_adj + c_trie_base + 1e-12)
#
#             # --- First misalignment override ---
#             if not is_first_misalignment and top_llm_token != top_trie_token:
#                 gamma = 1.0  # fully prefer LLM
#                 is_first_misalignment = True
#                 print(f"[FIRST MISMATCH @ step {i}] → Forcing LLM dominance (gamma=1.0)")
#
#             # --- Mix ---
#             p_mix = gamma * p_lm_temp + (1 - gamma) * p_trie
#             p_mix = p_mix / (p_mix.sum() + 1e-12)
#
#             top_mix_token = torch.argmax(p_mix).item()
#             top_mix_prob = p_mix[top_mix_token].item()
#
#             if top_mix_token == top_trie_token:
#                 last_matched_count[b] += 1
#             else:
#                 last_matched_count[b] = 0
#
#             if top_mix_token == eos_id: end_flag = True
#
#             print(f"c_lm_adj: {c_lm_adj:.3f} | gamma: {gamma:.3f} | D={D:.3f}")
#             print("SUMMARY LLM:", top_llm_token, Util.token_ids_to_text(torch.tensor([[top_llm_token]], device=logits.device), tokenizer), "(", top_llm_prob, ")")
#             print("SUMMARY TRIE:", top_trie_token, Util.token_ids_to_text(torch.tensor([[top_trie_token]], device=logits.device), tokenizer), "(", top_trie_score, ")")
#             print("SUMMARY MIX:", top_mix_token, Util.token_ids_to_text(torch.tensor([[top_mix_token]], device=logits.device), tokenizer), "(", top_mix_prob, ")")
#
#             print("----- ----- ---- --- ")
#
#             p_lm[b] = p_mix.clone()
#
#         idx_next = torch.argmax(p_lm, dim=-1, keepdim=True)
#
#         if end_flag:
#             idx_next = torch.full_like(idx_next, eos_id)
#
#         if eos_id is not None and (idx_next.squeeze(-1) == eos_id).all():
#             break
#         idx = torch.cat((idx, idx_next), dim=1)
#
#     print()
#     print("idx F:", idx, "Decoded F:", Util.token_ids_to_text(idx, tokenizer))
#     print("############################################################")
#     return idx


def generate_greedy1(model, tokenizer, batch, max_new_tokens, context_size, trie,
                     scoring_config=None, eos_id=None):
    scoring_args = {k: scoring_config[k]
                    for k in ["w_len", "w_freq", "w_recency"]
                    if k in scoring_config}
    last_matched_count = 0
    assert batch.shape[0] == 1, "generate_greedy1 currently supports only batch_size=1."

    # --- Placeholder & Trie tracking ---
    collecting_placeholder = False
    placeholder_seq = []
    current_trie_key = None

    for _ in range(max_new_tokens):
        # ---- Crop context ----
        batch_cond = batch[:, -context_size:]

        # ---- Forward pass ----
        with torch.no_grad():
            logits = model(batch_cond)  # (1, n_tokens, vocab_size)
        logits = logits[:, -1, :]      # (1, vocab_size)
        vocab_size = logits.shape[-1]

        # ---- Base LLM probabilities ----
        p_lm_base = torch.softmax(logits[0], dim=-1)
        final_p = p_lm_base  # may be mixed with trie later

        # ------------------------------------------------------------------
        # TRIE ENGAGEMENT for the *next* token (uses context of EMITTED tokens)
        # ------------------------------------------------------------------
        trie_ranked = []
        if current_trie_key is not None and len(current_trie_key) > 0:
            print(f"****** Active Trie key: {tokenizer.decode(current_trie_key)} ******")
            trie_ranked = trie.score_next_tokens(current_trie_key, **scoring_args)
        else:
            top_llm_token = torch.argmax(p_lm_base).item()
            token_text = Util.token_ids_to_text(
                torch.tensor([[top_llm_token]], device=logits.device), tokenizer
            )
            print(f"No Trie suggestions — using LLM base distribution.  Predicted token: {top_llm_token} {token_text}")

        if trie_ranked:
            print("****** Trie suggestions — using MIX distribution ******")
            top_trie_token, top_trie_score, _ = trie_ranked[0]
            trie_scores = torch.zeros(vocab_size, device=logits.device)
            for tok, score, _ in trie_ranked:
                trie_scores[tok] = max(score, 0.0)
            p_trie = (Util.normalize_trie_scores(trie_scores)
                      if trie_scores.max() > 0 else torch.zeros_like(trie_scores))

            c_llm = compute_llm_confidence(p_lm_base)
            T = compute_adaptive_temperature_exact(logits[0], top_trie_prob=top_trie_score)
            p_lm_temp = torch.softmax(logits[0] / T, dim=-1)
            print(f"Adaptive T (exact): {T:.4f}, target trie max={top_trie_score:.4f}, "
                  f"llm_max={p_lm_temp.max().item():.4f}")

            c_trie = float(p_trie.max().item()) if p_trie.sum() > 0 else 0.0
            print("LLM Confidence", c_llm)
            print("Trie Confidence", c_trie)

            D = compute_disagreement_D(p_lm_temp, p_trie, k=5)
            C = compute_continuity_C(last_matched_count)

            top_llm_token_temp = torch.argmax(p_lm_temp).item()
            top_llm_prob = p_lm_temp[top_llm_token_temp].item()

            c_lm_adj = c_llm * (1 - D ** 2)
            c_trie_adj = c_trie + (1 - c_trie) * (c_trie ** 2) * C
            print("LLM Confidence (After Penalizing)", c_lm_adj)
            print("Trie Confidence (BASE AS WELL)", c_trie_adj)

            gamma = c_lm_adj / (c_lm_adj + c_trie + 1e-12)
            p_mix = gamma * p_lm_temp + (1 - gamma) * p_trie
            p_mix = p_mix / (p_mix.sum() + 1e-12)

            top_mix_token = torch.argmax(p_mix).item()
            top_mix_prob = p_mix[top_mix_token].item()

            if top_mix_token == top_trie_token:
                last_matched_count += 1
            else:
                last_matched_count = 0

            print(f"c_lm_adj_confidence: {c_lm_adj:.3f} | gamma: {gamma:.3f} | D={D:.3f} | C={C:.3f}")
            print("SUMMARY LLM:", top_llm_token_temp,
                  Util.token_ids_to_text(torch.tensor([[top_llm_token_temp]], device=logits.device), tokenizer),
                  "(", top_llm_prob, ")")
            print("SUMMARY TRIE:", top_trie_token,
                  Util.token_ids_to_text(torch.tensor([[top_trie_token]], device=logits.device), tokenizer),
                  "(", top_trie_score, ")")
            print("SUMMARY MIX:", top_mix_token,
                  Util.token_ids_to_text(torch.tensor([[top_mix_token]], device=logits.device), tokenizer),
                  "(", top_mix_prob, ")")
            print("----- ----- ---- --- ")

            final_p = p_mix

        # ---- Decide the next token to EMIT (this is what gets appended) ----
        idx_next = torch.argmax(final_p, dim=-1).view(1, 1).to(batch.device)
        next_token = int(idx_next.item())



        # ------------------------------------------------------------------
        # UPDATE PLACEHOLDER/TRIE CONTEXT USING THE *EMITTED* TOKEN ONLY
        # ------------------------------------------------------------------
        if next_token == 22935:  # '{{'
            collecting_placeholder = True
            placeholder_seq = [next_token]
            # new placeholder → reset active key
            current_trie_key = None
            print(">> Started collecting placeholder sequence with '{{'")

        elif collecting_placeholder:
            placeholder_seq.append(next_token)
            if next_token == 11709:  # '}}' closes placeholder
                collecting_placeholder = False
                print(">> Completed placeholder:", tokenizer.decode(placeholder_seq))
                # Keep '}}' in the base key so the first suggestion is 'www'
                current_trie_key = placeholder_seq.copy()
                placeholder_seq = []
        else:
            # Not collecting a placeholder: extend the active trie key if one exists
            if current_trie_key is not None:
                # Keep growing with emitted tokens until a new '{{' starts
                # (we do NOT reset on punctuation anymore)
                if next_token != 22935:
                    current_trie_key.append(next_token)

        # ---- EOS check ----
        if eos_id is not None and (batch[:, -1] == eos_id).all():
            break

        # ---- Append emitted token to sequence ----
        batch = torch.cat((batch, idx_next), dim=1)
        # print("batch:", batch)

    return batch











def response_strategy_based_extractor_and_saver_abrupt_all_1(model, tokenizer, test_data, device, BASE_CONFIG, trie1, trie2, SCORING_CONFIG):
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
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ODD_1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        token_ids = generate_greedy1(
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
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ END ODD_1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")



        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("response=>", test_data[i]["response"])
    Util.store_in_json_file(test_data, Utils.Constants.RESPONSE_SPINE_FILE_NAME_ALL_1)



import re

def response_strategy_based_extractor_and_saver_abrupt1(model, tokenizer, test_data, device, BASE_CONFIG,
                                                        trie1, trie2,
                                                        SCORING_CONFIG):
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
        print("input_text:", input_text)
        print(".....................")
        print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ODD_{i} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        token_ids = generate_greedy1(
            model=model,
            tokenizer=tokenizer,
            batch=input_ids,
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            trie=trie,
            scoring_config=SCORING_CONFIG,
            eos_id=50256
        )

        print("Prompt length:", prompt_len,
              "Generated length:", token_ids.size(1),
              "Continuation length:", token_ids.size(1) - prompt_len)

        # Decode only continuation
        continuation_token_ids = token_ids[:, prompt_len:]
        response_text = Util.token_ids_to_text(continuation_token_ids, tokenizer)

        # --- Clean model output: remove placeholders + markers ---
        response_text = re.sub(
            r"\{\{[A-Z0-9_]+\}\}",
            "",
            response_text.replace("### Response:", "")
                         .replace("<|endoftext|>", "")
                         .strip()
        ).strip()
        print("model response=>", response_text)

        # --- Clean ground truth response as well ---
        test_data[i]["response"] = re.sub(r"\{\{[A-Z0-9_]+\}\}", "", test_data[i]["response"]).strip()

        # --- Save model response ---
        test_data[i]["model_response"] = response_text

        print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ END ODD_{i} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("response=>", test_data[i]["response"])

    Util.store_in_json_file(test_data, Utils.Constants.RESPONSE_SPINE_FILE_NAME)


def response_strategy_based_extractor_and_saver_incremental(model, tokenizer, test_data, device, BASE_CONFIG,
                                                        trie1, trie2, trie3, trie4, trie5,
                                                        SCORING_CONFIG):
    n_test = len(test_data)
    portion = n_test // 5

    test_data = list(test_data)  # Convert HF dataset to a list of dicts
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        input_text = AplacaFineTuner.format_input(entry)
        input_ids = Util.text_to_token_ids(input_text, tokenizer).to(device)
        prompt_len = input_ids.size(1)

        # ---- Select corresponding TRIE ----
        if i < portion:
            trie = trie1
            print("TRIE is TRIE1 (P1)")
        elif i < 2 * portion:
            trie = trie2
            print("TRIE is TRIE2 (P2)")
        elif i < 3 * portion:
            trie = trie3
            print("TRIE is TRIE3 (P3)")
        elif i < 4 * portion:
            trie = trie4
            print("TRIE is TRIE4 (P4)")
        else:
            trie = trie5
            print("TRIE is TRIE5 (P5)")

        print(".....................")
        print("input_text:", input_text)
        print(".....................")
        print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ODD_{i} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        token_ids = generate_greedy1(
            model=model,
            tokenizer=tokenizer,
            batch=input_ids,
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            trie=trie,
            scoring_config=SCORING_CONFIG,
            eos_id=50256
        )

        print("Prompt length:", prompt_len,
              "Generated length:", token_ids.size(1),
              "Continuation length:", token_ids.size(1) - prompt_len)

        # Decode only continuation
        continuation_token_ids = token_ids[:, prompt_len:]
        response_text = Util.token_ids_to_text(continuation_token_ids, tokenizer)

        # --- Clean model output: remove placeholders + markers ---
        response_text = re.sub(
            r"\{\{[A-Z0-9_]+\}\}",
            "",
            response_text.replace("### Response:", "")
                         .replace("<|endoftext|>", "")
                         .strip()
        ).strip()
        print("model response=>", response_text)

        # --- Clean ground truth response as well ---
        test_data[i]["response"] = re.sub(r"\{\{[A-Z0-9_]+\}\}", "", test_data[i]["response"]).strip()

        # --- Save model response ---
        test_data[i]["model_response"] = response_text

        print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ END ODD_{i} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("response=>", test_data[i]["response"])

    Util.store_in_json_file(test_data, Utils.Constants.RESPONSE_SPINE_FILE_NAME)



def response_strategy_based_extractor_and_saver_gradual(model, tokenizer, test_data, device, BASE_CONFIG,
                                                        trie1, trie2, trie3, trie4, trie5, trie6,
                                                        SCORING_CONFIG):
    n_test = len(test_data)
    portion = n_test // 6

    test_data = list(test_data)  # Convert HF dataset to a list of dicts
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        input_text = AplacaFineTuner.format_input(entry)
        input_ids = Util.text_to_token_ids(input_text, tokenizer).to(device)
        prompt_len = input_ids.size(1)

        # ---- Select corresponding TRIE ----
        if i < portion:
            trie = trie1
            print("TRIE is TRIE1 (P1)")
        elif i < 2 * portion:
            trie = trie2
            print("TRIE is TRIE2 (P2)")
        elif i < 3 * portion:
            trie = trie3
            print("TRIE is TRIE3 (P3)")
        elif i < 4 * portion:
            trie = trie4
            print("TRIE is TRIE4 (P4)")
        elif i < 5 * portion:
            trie = trie5
            print("TRIE is TRIE5 (P5)")
        else:
            trie = trie6
            print("TRIE is TRIE5 (P6)")

        print(".....................")
        print("input_text:", input_text)
        print(".....................")
        print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ODD_{i} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        token_ids = generate_greedy1(
            model=model,
            tokenizer=tokenizer,
            batch=input_ids,
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            trie=trie,
            scoring_config=SCORING_CONFIG,
            eos_id=50256
        )

        print("Prompt length:", prompt_len,
              "Generated length:", token_ids.size(1),
              "Continuation length:", token_ids.size(1) - prompt_len)

        # Decode only continuation
        continuation_token_ids = token_ids[:, prompt_len:]
        response_text = Util.token_ids_to_text(continuation_token_ids, tokenizer)

        # --- Clean model output: remove placeholders + markers ---
        response_text = re.sub(
            r"\{\{[A-Z0-9_]+\}\}",
            "",
            response_text.replace("### Response:", "")
                         .replace("<|endoftext|>", "")
                         .strip()
        ).strip()
        print("model response=>", response_text)

        # --- Clean ground truth response as well ---
        test_data[i]["response"] = re.sub(r"\{\{[A-Z0-9_]+\}\}", "", test_data[i]["response"]).strip()

        # --- Save model response ---
        test_data[i]["model_response"] = response_text

        print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ END ODD_{i} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("response=>", test_data[i]["response"])

    Util.store_in_json_file(test_data, Utils.Constants.RESPONSE_SPINE_FILE_NAME)

if __name__ == "__main__":
    print("\n--- Test Variant #1: No C at all ---\n")

    def test_case(p_lm, p_trie):
        D = compute_disagreement_D(p_lm, p_trie, k=5)
        c_lm = compute_llm_confidence(p_lm)
        c_trie = float(p_trie.max().item())

        c_lm_adj = c_lm * (1 - D**2)
        gamma = c_lm_adj / (c_lm_adj + c_trie + 1e-12)

        top_llm = torch.argmax(p_lm).item()
        top_trie = torch.argmax(p_trie).item()
        top_mix = torch.argmax(gamma * p_lm + (1 - gamma) * p_trie).item()

        print(f"p_lm={p_lm.tolist()} | p_trie={p_trie.tolist()}")
        print(f"c_lm={c_lm:.3f}, c_trie={c_trie:.3f}, D={D:.3f}")
        print(f"c_lm_adj={c_lm_adj:.3f}, gamma={gamma:.3f}")
        print(f"Top LLM={top_llm}, Top Trie={top_trie} → Winner: {'LLM' if top_mix == top_llm else 'Trie'}\n")

    # Run test cases
    test_case(torch.tensor([0.97, 0.02, 0.01]), torch.tensor([0.4, 0.3, 0.3]))
    test_case(torch.tensor([0.6, 0.3, 0.1]), torch.tensor([0.3, 0.6, 0.1]))
    test_case(torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0.7, 0.2, 0.1]))
    test_case(torch.tensor([0.9, 0.05, 0.05]), torch.tensor([0.05, 0.9, 0.05]))
