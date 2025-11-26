import Utils.Constants
from Utils import Util
from DataFineTuning import AplacaFineTuner
from tqdm import tqdm
import re

def generate(model, batch, max_new_tokens, context_size, eos_id=None):
    # idx is (batch, n_tokens) array of indices in the current context

    ###Input batch:
    ###tensor([[6109, 3626, 6100,  345],
    ##         [6109, 1110, 6622,  257]]
    #        )

    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        batch_cond = batch[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(batch_cond)  ### batch, n_tokens, vocab_size

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size) -since we only used the last word corr vector in the batch
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # # Early stop if all just produced EOS
        # if eos_id is not None and torch.all(idx_next.squeeze(-1) == eos_id):
        #     break

        # when the statement ends with all of <eos> stop
        if eos_id is not None and (batch[:, -1] == eos_id).all():
            break

        # Append sampled index to the running sequence
        batch = torch.cat((batch, idx_next), dim=1)  # (batch, n_tokens+1)

    return batch

import torch

# def generate(model, batch, max_new_tokens, context_size, eos_id=None):
#
#     # Preserve model mode
#     was_training = model.training
#     model.eval()
#
#     # Track EOS per sequence
#     eos_mask = torch.zeros(batch.size(0), dtype=torch.bool, device=batch.device)
#
#     for _ in range(max_new_tokens):
#         # Truncate context
#         batch_cond = batch[:, -context_size:]
#
#         # Forward pass: expect raw logits tensor (B, T, V)
#         with torch.no_grad():
#             logits = model(batch_cond)
#
#         # If your model returns extra items, keep only the first (logits)
#         if isinstance(logits, (tuple, list)):
#             logits = logits[0]
#
#         # Focus on the last time step: (B, V)
#         logits = logits[:, -1, :]
#
#         # Greedy pick (argmax over logits; softmax not needed)
#         idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
#
#         # Mark EOS per sequence
#         if eos_id is not None:
#             eos_mask |= (idx_next.squeeze(1) == eos_id)
#
#         # Append to running sequence
#         batch = torch.cat((batch, idx_next), dim=1)  # (B, current_len+1)
#
#         # Early stop: all sequences finished
#         if eos_id is not None and eos_mask.all():
#             break
#
#
#     # Restore mode
#     if was_training:
#         model.train()
#
#     return batch

import torch
from torch.nn.utils.rnn import pad_sequence

# def generate(model, batch, max_new_tokens, context_size, eos_id=None):
#     """
#     Generates text using greedy decoding with context truncation and EOS-aware stopping.
#
#     Args:
#         model: The language model to use for generation.
#         batch: Input tensor of shape (B, T) where B = batch size, T = initial token count.
#         max_new_tokens: Maximum number of new tokens to generate.
#         context_size: Number of tokens to keep in context (for memory efficiency).
#         eos_id: ID of the end-of-sequence token. If None, generation runs for all steps.
#
#     Returns:
#         A tensor of shape (B, T') with generated token sequences (padded if needed).
#     """
#     # Preserve original mode
#     was_training = model.training
#     model.eval()
#
#     B = batch.size(0)
#     device = batch.device
#
#     # Track EOS state per sequence
#     eos_mask = torch.zeros(B, dtype=torch.bool, device=device) if eos_id is not None else None
#
#     for _ in range(max_new_tokens):
#         # Truncate to context window
#         batch_cond = batch[:, -context_size:]
#
#         with torch.no_grad():
#             logits = model(batch_cond)
#             if isinstance(logits, (tuple, list)):
#                 logits = logits[0]  # Only keep logits if model returns extra values
#
#         logits = logits[:, -1, :]  # (B, V)
#
#         # Pick next token (greedy)
#         idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
#
#
#
#         if eos_id is not None:
#             # Only consider EOS after at least one new token has been generated
#             if batch.size(1) > batch_cond.size(1):
#                 just_eos = (idx_next.squeeze(1) == eos_id)
#                 eos_mask |= just_eos
#
#             # For finished sequences, just keep EOS token
#             idx_next[eos_mask] = eos_id
#
#
#         # Append next token
#         batch = torch.cat((batch, idx_next), dim=1)
#
#         # Early stop: all finished
#         if eos_id is not None and eos_mask.all():
#             break
#
#     # Restore training mode if necessary
#     if was_training:
#         model.train()
#
#     # Post-process: remove tokens after EOS
#     if eos_id is not None:
#         cleaned_batch_list = []
#         for i in range(B):
#             eos_pos = (batch[i] == eos_id).nonzero(as_tuple=False)
#             if eos_pos.numel() > 0:
#                 end_idx = eos_pos[0].item()
#                 if end_idx == 0:
#                     # Keep at least the first token if EOS appears at start
#                     cleaned_batch_list.append(batch[i, :1])
#                 else:
#                     cleaned_batch_list.append(batch[i, :end_idx+1])
#             else:
#                 cleaned_batch_list.append(batch[i])
#
#         cleaned_batch = pad_sequence(
#             cleaned_batch_list, batch_first=True, padding_value=eos_id if eos_id else 50256
#         )
#
#         return cleaned_batch
#
#     return batch

import re

def response_strategy_based_extractor_and_saver(model, tokenizer, test_data, device, BASE_CONFIG):
    print("########################### GREEDY.py ###########################")
    test_data = list(test_data)  # Convert HF dataset to a list of dicts

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = AplacaFineTuner.format_input(entry)
        input_ids = Util.text_to_token_ids(input_text, tokenizer).to(device)
        prompt_len = input_ids.size(1)

        token_ids = generate(
            model=model,
            batch=input_ids,
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )

        print("Prompt length:", prompt_len,
              "Generated length:", token_ids.size(1),
              "Continuation length:", token_ids.size(1) - prompt_len)

        # Decode only continuation
        continuation_tokens = token_ids[:, prompt_len:]
        response_text = Util.token_ids_to_text(continuation_tokens, tokenizer)

        # --- Clean model response (remove placeholders + special markers) ---
        response_text = re.sub(
            r"\{\{[A-Z0-9_]+\}\}",
            "",
            response_text.replace("### Response:", "")
                         .replace("<|endoftext|>", "")
                         .strip()
        ).strip()

        # Apply any extra cleaning from Util
        response_text = Util.clean_generated_text(response_text)

        # --- Clean ground-truth response too ---
        test_data[i]["response"] = re.sub(r"\{\{[A-Z0-9_]+\}\}", "", test_data[i]["response"]).strip()

        # --- Save cleaned response ---
        test_data[i]["model_response"] = response_text

    Util.store_in_json_file(test_data, Utils.Constants.RESPONSE_GREEDY_FILE_NAME)





if __name__ == "__main__":

    response_text = """If you have noticed discrepancies in your bill and wish to contest them, please adhere to the following steps:

    1. Log in to your account on {{WEBSITE_URL}} www.globaltel-communications.com.
    2. Navigate to the {{INVOICE_SECTION}} Detailed Invoice History and Comprehensive Billing Archive section.
    3. Select the bill you wish to dispute.
    4. Click on the {{DISPUTE_INVOICE_OPTION}} Initiate Formal Invoice Dispute and Investigation Request to dispute the charge.
    5. Fill in the required information and submit your dispute.

    Our team will review your submission and get back to you within {{DAYS_NUMBER}} 5 Business Days (Processing Timeframe) business days.
    """

    # --- Remove only placeholders {{...}}
    response_text = re.sub(r"\{\{[A-Za-z0-9_]+\}\}", "", response_text)

    # --- Normalize spaces and clean formatting
    response_text = re.sub(r"\s+", " ", response_text).strip()

    print(response_text)


