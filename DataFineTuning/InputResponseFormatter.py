import torch
from DataFineTuning import AplacaFineTuner


"""
Custom collate function that prepares inputs and targets for next-token prediction tasks.

Steps:
1. Determine the maximum sequence length in the batch and add +1 to account for the EOT token.
2. For each sequence in the batch:
   a. Append the EOT token (pad_token_id) to the end of the sequence.
   b. Pad the sequence with pad_token_id to ensure equal length across the batch.
   c. Create the input sequence by removing the last token (for causal modeling).
   d. Create the target sequence by removing the first token (next-token prediction target).
   e. Replace all but the first occurrence of pad_token_id in the target with `ignore_index` (default: -100),
      so that the loss is not computed on padded positions.
   f. If allowed_max_length is set, truncate both input and target sequences to this length.
3. Stack all input and target sequences into tensors.
4. Move both tensors to the specified device.
5. Return a tuple: (inputs_tensor, targets_tensor).
"""

def collate_input_targets_with_end_token_and_pad(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        # indices = torch.nonzero(mask).squeeze()
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor



def dictbatch_to_tokenlists_then_collate(batch, tokenizer, raw_collate):
    """Takes a batch of dicts, builds a single string per item, encodes to ids,
    then calls your original collate which expects List[int]."""
    seqs = []
    for item in batch:
        # If you already precomputed ids in the item, prefer them
        if "input_ids" in item and isinstance(item["input_ids"], list):
            seqs.append(item["input_ids"])
            continue

        # Otherwise, build the prompt for next-token prediction
        # Adjust if your formatting function is different
        text = AplacaFineTuner.format_input(item)  # uses your fields
        ids = tokenizer.encode(text)              # tiktoken -> List[int]
        seqs.append(ids)

    # Safety: your collate expects lists
    assert all(isinstance(x, list) for x in seqs), "Batch items must be lists"
    return raw_collate(seqs)


# if __name__ == "__main__":
#     inputs_1 = [0, 1, 2, 3, 4]
#     inputs_2 = [5, 6]
#     inputs_3 = [7, 8, 9]
#
#     batch = (
#         inputs_1,
#         inputs_2,
#         inputs_3
#     )
#     inputs, targets  = collate_input_targets_with_end_token_and_pad(batch)
#     print("inputs", inputs)
#     print("targets", targets)

import torch
from DataFineTuning import InputResponseFormatter


def test_collate_cases():
    print("=== Collate Function Tests ===")

    # Case 1 - unequal length
    inputs_1 = [1, 2]
    inputs_2 = [4, 5]
    inputs_3 = [6]
    batch = (inputs_1, inputs_2, inputs_3)
    inputs, targets = InputResponseFormatter.collate_input_targets_with_end_token_and_pad(batch)
    print("Case 1 - unequal length")
    print("inputs:\n", inputs)
    print("targets:\n", targets)

    # Case 2 - truncated to 3
    inputs_1 = [1, 2, 3, 4]
    inputs_2 = [4, 5]
    inputs_3 = [6]
    batch = (inputs_1, inputs_2, inputs_3)
    inputs, targets = InputResponseFormatter.collate_input_targets_with_end_token_and_pad(batch, allowed_max_length=3)
    print("Case 2 - truncated to 3")
    print("inputs:\n", inputs)
    print("targets:\n", targets)

    # Case 3 - pad masking
    inputs_1 = [1, 2, 3]
    inputs_2 = [3, 4, 5]
    batch = (inputs_1, inputs_2)
    inputs, targets = InputResponseFormatter.collate_input_targets_with_end_token_and_pad(batch)
    print("Case 3 - pad masking")
    print("targets:\n", targets)


def test_nonzero_flatten_cases():
    print("\n=== Nonzero().flatten() Tests ===")

    # Case 1: Multiple pad positions
    mask = torch.tensor([False, True, True, False])
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    print("Case 1 - multiple pads:", indices.tolist())  # expect [1, 2]

    # Case 2: Single pad position
    mask = torch.tensor([False, True, False, False])
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    print("Case 2 - single pad:", indices.tolist())  # expect [1]

    # Case 3: No pad positions
    mask = torch.tensor([False, False, False, False])
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    print("Case 3 - no pads:", indices.tolist())  # expect []

    # Case 4: All pad positions
    mask = torch.tensor([True, True, True, True])
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    print("Case 4 - all pads:", indices.tolist())  # expect [0, 1, 2, 3]

    # Case 5: Batch-like (2D) mask
    mask = torch.tensor([[False, True, False],
                         [True, False, True]])
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    print("Case 5 - 2D mask flattened:", indices.tolist())
    # expect [1, 3, 5]


if __name__ == "__main__":
    test_collate_cases()
    test_nonzero_flatten_cases()

