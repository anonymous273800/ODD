import torch
import torch

import Utils.Constants
from Utils import Util
from tqdm import tqdm
from Utils import Util
from DataFineTuning import AplacaFineTuner
from tqdm import tqdm

# def generate(model, idx, max_new_tokens, context_size, eos_id=None, temperature=1.0, top_k=20):
#     # For-loop is the same as before: Get logits, and only focus on last time step
#     for _ in range(max_new_tokens):
#         idx_cond = idx[:, -context_size:]
#         with torch.no_grad():
#             logits = model(idx_cond)
#
#         logits = logits[:, -1, :]
#
#         # New: Filter logits with top_k sampling
#         if top_k is not None:
#             # Keep only top_k values
#             top_logits, _ = torch.topk(logits, top_k)
#             min_val = top_logits[:, -1]
#             logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
#
#         # New: Apply temperature scaling
#         if temperature > 0.0:
#             logits = logits / temperature
#
#             # Apply softmax to get probabilities
#             probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
#
#             # Sample from the distribution
#             idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
#
#         # Otherwise same as before: get idx of the vocab entry with the highest logits value
#         else:
#             idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
#
#         if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
#             break
#
#         # Same as before: append sampled index to the running sequence
#         idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)
#
#     return idx

def generate(model, idx, max_new_tokens, context_size, eos_id=None, temperature=1.0, top_k=20):
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]


        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
        #     break
        # when the statement ends with all of <eos> stop
        if eos_id is not None and (idx[:, -1] == eos_id).all():
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


import torch

# def generate(model, idx, max_new_tokens, context_size, eos_id=None, temperature=1.0, top_k=20):
#     was_training = model.training
#     model.eval()
#
#
#     for _ in range(max_new_tokens):
#         idx_cond = idx[:, -context_size:]
#
#         with torch.no_grad():
#             logits = model(idx_cond)
#         if isinstance(logits, (tuple, list)):
#             logits = logits[0]
#
#         # (B, V) last-step logits
#         logits = logits[:, -1, :]
#
#
#         # 2) Top-k filtering (before temperature and softmax)
#         if top_k is not None and top_k > 0:
#             k = min(top_k, logits.size(-1))
#             topk_vals, _ = torch.topk(logits, k, dim=-1)
#             thresh = topk_vals[:, -1].unsqueeze(-1)  # (B, 1)
#             neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
#             logits = torch.where(logits < thresh, neg_inf, logits)
#
#         # 3) Sampling with temperature or greedy
#         if temperature is not None and temperature > 0.0:
#             logits = logits / temperature
#             probs = torch.softmax(logits, dim=-1)  # (B, V)
#             # Numerical guard: replace any NaNs (can happen if all -inf) with uniform
#             if torch.isnan(probs).any():
#                 V = probs.size(-1)
#                 probs = torch.full_like(probs, 1.0 / V)
#             idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
#         else:
#             # Greedy
#             idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
#
#
#         # 4) Append
#         idx = torch.cat((idx, idx_next), dim=1)
#
#
#     if was_training:
#         model.train()
#
#     return idx



import torch
from torch.nn.utils.rnn import pad_sequence

# def generate(model, idx, max_new_tokens, context_size, eos_id=None, temperature=1.0, top_k=20):
#     """
#     Generates tokens using top-k sampling with temperature scaling and robust EOS handling.
#
#     Args:
#         model: Language model.
#         idx: Tensor of shape (B, T) with input token IDs.
#         max_new_tokens: Number of tokens to generate.
#         context_size: Max context window.
#         eos_id: End-of-sequence token ID. If None, generation continues for all steps.
#         temperature: Scaling for logits.
#         top_k: Top-k filtering for sampling.
#
#     Returns:
#         Tensor (B, T') of generated sequences, cleaned up after EOS or placeholder.
#     """
#     was_training = model.training
#     model.eval()
#
#     B = idx.size(0)
#     device = idx.device
#
#     eos_mask = torch.zeros(B, dtype=torch.bool, device=device) if eos_id is not None else None
#
#     for _ in range(max_new_tokens):
#         idx_cond = idx[:, -context_size:]  # truncate context
#
#         with torch.no_grad():
#             logits = model(idx_cond)
#         if isinstance(logits, (tuple, list)):
#             logits = logits[0]
#
#         logits = logits[:, -1, :]  # (B, V)
#
#         # 1) Top-k filtering
#         if top_k is not None and top_k > 0:
#             k = min(top_k, logits.size(-1))
#             topk_vals, _ = torch.topk(logits, k, dim=-1)
#             thresh = topk_vals[:, -1].unsqueeze(-1)  # (B, 1)
#             neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
#             logits = torch.where(logits < thresh, neg_inf, logits)
#
#         # 2) Apply temperature and sample
#         if temperature is not None and temperature > 0.0:
#             logits = logits / temperature
#             probs = torch.softmax(logits, dim=-1)
#
#             # Numerical fallback in case of NaNs (e.g. if all logits = -inf)
#             if torch.isnan(probs).any():
#                 V = probs.size(-1)
#                 probs = torch.full_like(probs, 1.0 / V)
#
#             idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
#         else:
#             # Greedy fallback
#             idx_next = torch.argmax(logits, dim=-1, keepdim=True)
#
#         # # --- EOS handling ---
#         if eos_id is not None:
#             # Only consider EOS after at least one new token has been generated
#             if idx.size(1) > idx_cond.size(1):
#                 just_eos = (idx_next.squeeze(1) == eos_id)
#                 eos_mask |= just_eos
#
#             # For finished sequences, just keep EOS token
#             idx_next[eos_mask] = eos_id
#
#         # 4) Append to sequence
#         idx = torch.cat((idx, idx_next), dim=1)
#
#         # 5) Early stop
#         if eos_id is not None and eos_mask.all():
#             break
#
#     if was_training:
#         model.train()
#
#     # Post-process: remove tokens after EOS or -1
#     if eos_id is not None:
#         cleaned_batch_list = []
#         for i in range(B):
#             eos_pos = (idx[i] == eos_id).nonzero(as_tuple=False)
#             if eos_pos.numel() > 0:
#                 end_idx = eos_pos[0].item()
#                 if end_idx == 0:
#                     # Keep at least the first token if EOS appears at start
#                     cleaned_batch_list.append(idx[i, :1])
#                 else:
#                     cleaned_batch_list.append(idx[i, :end_idx+1])
#             else:
#                 cleaned_batch_list.append(idx[i])
#
#         cleaned_batch = pad_sequence(
#             cleaned_batch_list, batch_first=True, padding_value=eos_id if eos_id else 50256
#         )
#         return cleaned_batch
#
#     return idx


import re

def response_strategy_based_extractor_and_saver(model, tokenizer, test_data, device, BASE_CONFIG):
    test_data = list(test_data)  # Convert HF dataset to a list of dicts
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = AplacaFineTuner.format_input(entry)
        input_ids = Util.text_to_token_ids(input_text, tokenizer).to(device)
        prompt_len = input_ids.size(1)

        token_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
            temperature=1.0,
            top_k=5
        )

        print("Prompt length:", prompt_len,
              "Generated length:", token_ids.size(1),
              "Continuation length:", token_ids.size(1) - prompt_len)

        # Decode only continuation
        continuation_tokens = token_ids[:, prompt_len:]
        response_text = Util.token_ids_to_text(continuation_tokens, tokenizer)

        # --- Clean model response: remove placeholders + special markers ---
        response_text = re.sub(
            r"\{\{[A-Z0-9_]+\}\}",
            "",
            response_text.replace("### Response:", "")
                         .replace("<|endoftext|>", "")
                         .strip()
        ).strip()

        # Apply any additional cleaning
        response_text = Util.clean_generated_text(response_text)

        # --- Clean ground truth response as well ---
        test_data[i]["response"] = re.sub(r"\{\{[A-Z0-9_]+\}\}", "", test_data[i]["response"]).strip()

        # --- Save cleaned model response ---
        test_data[i]["model_response"] = response_text

    Util.store_in_json_file(test_data, Utils.Constants.RESPONSE_TOPK_TEMPSCALING_FILE_NAME)
