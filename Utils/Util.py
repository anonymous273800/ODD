import torch
import json
import re
from Utils import Constants
import os


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())



# def store_in_json_file(data, file_name):
#     # Convert Hugging Face Dataset to list of row dicts
#     if hasattr(data, "to_dict"):
#         column_dict = data.to_dict()  # {column_name: [val1, val2, ...]}
#         data = [dict(zip(column_dict.keys(), row)) for row in zip(*column_dict.values())]
#
#     with open(file_name, "w", encoding="utf-8") as file:
#         json.dump(data, file, indent=4, ensure_ascii=False)


def store_in_json_file(data, file_name):
    """
    Stores data to a JSON file under the directory specified by PREDICTIONS_PATH.
    Converts Hugging Face Datasets to list of dicts if needed.
    """
    if hasattr(data, "to_dict"):
        column_dict = data.to_dict()
        data = [dict(zip(column_dict.keys(), row)) for row in zip(*column_dict.values())]

    # Ensure directory exists
    os.makedirs(Constants.PREDICTIONS_PATH, exist_ok=True)

    # Build full file path
    full_path = os.path.join(Constants.PREDICTIONS_PATH, file_name)

    with open(full_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f" Saved to: {full_path}")



def save_model(model, CHOOSE_MODEL, NUM_EPOCHS):
    clean_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
    file_name = f"{clean_name}-epochs-{NUM_EPOCHS}-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")

def load_model(model, checkpoint_path, device):
    # "gpt2-medium355M-sft.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model



def load_json_file(file_name, base_path=None):
    """
    Loads a JSON file from the specified base path.
    If base_path is not provided, it defaults to Constants.PREDICTIONS_PATH.
    """
    if base_path is not None:
        full_path = os.path.join(base_path, Constants.PREDICTIONS_PATH, file_name)
    else:
        full_path = os.path.join(Constants.PREDICTIONS_PATH, file_name)


    with open(full_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def load_json_file2(file_name):

    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# def decode_token_ids(tokenizer, ids):
#     toks = tokenizer.convert_ids_to_tokens(ids)
#     # convert_tokens_to_string keeps spacing rules (e.g., leading space for Ġ tokens)
#     return [tokenizer.convert_tokens_to_string([t]) for t in toks]


def normalize_trie_scores(trie_scores):
    probs = torch.zeros_like(trie_scores)
    top_tok = torch.argmax(trie_scores)
    top_val = trie_scores[top_tok].item()

    rest = trie_scores.clone()
    rest[top_tok] = 0.0
    rest_sum = rest.sum()

    if rest_sum > 0:
        # Assign top token its raw score, scale others proportionally
        probs[top_tok] = top_val
        probs[rest > 0] = (1.0 - top_val) * rest[rest > 0] / rest_sum
    else:
        probs[top_tok] = 1.0
    return probs


def print_trie_score(trie_scores, p_trie, tokenizer):
    # Debug: print top-k tokens from both raw and normalized
    topk_raw = torch.topk(trie_scores, 5)  # original raw scores
    topk_norm = torch.topk(p_trie, 5)  # normalized distribution

    print("\n--- Trie Debug ---")
    print("Raw top-5:")
    for tok, val in zip(topk_raw.indices, topk_raw.values):
        print(f"  ID {tok.item()} | {tokenizer.decode([tok.item()])} | raw={val.item():.4f}")

    print("Normalized top-5:")
    for tok, val in zip(topk_norm.indices, topk_norm.values):
        print(f"  ID {tok.item()} | {tokenizer.decode([tok.item()])} | prob={val.item():.4f}")



def clean_generated_text(response_text):
    # remove placeholders
    # --- Remove only placeholders {{...}}
    response_text = re.sub(r"\{\{[A-Za-z0-9_]+\}\}", "", response_text)
    # --- Normalize spaces and clean formatting
    response_text = re.sub(r"\s+", " ", response_text).strip()
    return response_text

if __name__ == "__main__":
    base_path = "/Experiments/Regular001/Experiment001"
    data = load_json_file("response_greedy.json", base_path)
    print(data)