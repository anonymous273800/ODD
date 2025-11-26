import torch
from torch.utils.data import Dataset, DataLoader
from DataFineTuning import AplacaFineTuner

# --- Dataloader setup (force pinning OFF) ---
def make_loader(ds, batch_size, shuffle, collate_fn):
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,  # same as before
        num_workers=0,  # IMPORTANT: CUDA in collate -> keep 0
        pin_memory=False,  # HARD-OFF
        pin_memory_device="",  # HARD-OFF (empty string)
        collate_fn=collate_fn,
        persistent_workers=False  # be explicit since workers=0
    )
    # Sanity check: print the pin settings so we know it’s really off
    print(
        f"Loader pin_memory={loader.pin_memory}, "
        f"pin_memory_device={repr(loader.pin_memory_device)}"
    )
    return loader

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = AplacaFineTuner.format_input(entry)
            response_text = f"\n\n### Response:\n{entry['response']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)