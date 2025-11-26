import time
from functools import partial

import tiktoken
import torch
from datasets import Dataset, concatenate_datasets

from DataFineTuning import InputResponseFormatter
from DataLoading import DataLoading
from DataLoading.DataLoading import InstructionDataset
from Datasets import BitextTelecoDS
from Finetuning import Trainer
from LossHandler import LossCalculator
from Models.LLM.GPT2 import GPTModel
from PreTrainedGPTLoader import gpt_download3, gpt_loader
from Utils import Util

if __name__ == "__main__":
    # 1. set seed
    torch.manual_seed(42)

    # 2. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU Name:", torch.cuda.get_device_name(0))


    # 3. Load Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 4. Load the Dataset - Train, Test, Validation
    train_ratio=0.6
    test_ratio = 0.3
    val_ratio = 0.1
    dataset = BitextTelecoDS.get_bitext_telecom_dataset_splits(train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)
    train_data = dataset["train"]
    test_data = dataset["test"]
    val_data = dataset["validation"]
    combined_data = dataset["combined"]

    # # #todo: remove this on production
    # train_data = train_data.select(range(8))
    # test_data = test_data.select(range(8))
    # val_data = val_data.select(range(8))
    # combined_data = concatenate_datasets([train_data, test_data])

    train_data_p1 = BitextTelecoDS.preprocess_bitext_ds2(train_data, "p1", "gradual")
    val_data_p1 = BitextTelecoDS.preprocess_bitext_ds2(val_data, "p1", "gradual")

    # Split test data into 6 Gradual portions (p1 to p5)
    n_test = len(test_data)
    portion = n_test // 6

    test_data_p1 = BitextTelecoDS.preprocess_bitext_ds2(
        test_data.select(range(0, portion)),
        placeholder="p1",
        drift_type="gradual"
    )

    test_data_p2 = BitextTelecoDS.preprocess_bitext_ds2(
        test_data.select(range(portion, 2 * portion)),
        placeholder="p2",
        drift_type="gradual"
    )

    test_data_p3 = BitextTelecoDS.preprocess_bitext_ds2(
        test_data.select(range(2 * portion, 3 * portion)),
        placeholder="p3",
        drift_type="gradual"
    )

    test_data_p4 = BitextTelecoDS.preprocess_bitext_ds2(
        test_data.select(range(3 * portion, 4 * portion)),
        placeholder="p4",
        drift_type="gradual"
    )

    test_data_p5 = BitextTelecoDS.preprocess_bitext_ds2(
        test_data.select(range(4 * portion, 5 * portion)),
        placeholder="p5",
        drift_type="gradual"
    )

    test_data_p6 = BitextTelecoDS.preprocess_bitext_ds2(
        test_data.select(range(5 * portion, n_test)),
        placeholder="p6",
        drift_type="gradual"
    )

    # Convert your preprocessed lists to Dataset
    test_data_p1 = Dataset.from_list(test_data_p1)
    test_data_p2 = Dataset.from_list(test_data_p2)
    test_data_p3 = Dataset.from_list(test_data_p3)
    test_data_p4 = Dataset.from_list(test_data_p4)
    test_data_p5 = Dataset.from_list(test_data_p5)
    test_data_p6 = Dataset.from_list(test_data_p6)

    # Merge them back
    test_data_p1p6 = concatenate_datasets([test_data_p1, test_data_p2, test_data_p3, test_data_p4, test_data_p5, test_data_p5])  # preserve their sequence.


    # 5. Data Loaders
    raw_collate = partial(
        InputResponseFormatter.collate_input_targets_with_end_token_and_pad,
        device=device,
        allowed_max_length=1024,
    )

    train_dataset_p1 = InstructionDataset(train_data_p1, tokenizer)
    val_dataset_p1 = InstructionDataset(val_data_p1, tokenizer)
    test_dataset_p1p6 = InstructionDataset(test_data_p1p6, tokenizer)
    train_loader_p1 = DataLoading.make_loader(train_dataset_p1, batch_size=8, shuffle=True, collate_fn=raw_collate)
    val_loader_p1 = DataLoading.make_loader(val_dataset_p1, batch_size=8, shuffle=False, collate_fn=raw_collate)
    test_loader_p1p6 = DataLoading.make_loader(test_dataset_p1p6, batch_size=8, shuffle=False, collate_fn=raw_collate)


    # 4. Call LLM - GPT2 - Pretrained Weights
    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = gpt_download3.download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )
    model = GPTModel.GPTModel(BASE_CONFIG)
    gpt_loader.load_weights_into_gpt(model, params)
    model.to(device)
    model.eval()

    # 5. Before (Fine Tuning) Compute Loss. Loss Computation on Pretrained Model GPT-2
    print("--- ---- ---- ---- ---- ---- ")
    print("Loss Computation on the Pre-trained Model, before fine-tuning")
    LossCalculator.print_model_loss_for_all_datasets(model, train_loader_p1, test_loader_p1p6, val_loader_p1, device)
    print("--- ---- ---- ---- ---- ---- ")

    # 6. Regular General Finetuning (Training) on Dataset
    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    NUM_EPOCHS = 10
    model, train_losses, val_losses, tokens_seen = Trainer.train_model_simple(
        model, train_loader_p1, val_loader_p1, optimizer, device,
        num_epochs=NUM_EPOCHS, eval_freq=5, eval_iter=5
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    print("--- ---- ---- ---- ---- ---- ")
    # 5. Optional - After Training (Fine Tuning) Compute Loss.
    print("Loss Computation on the Finetuned Model (trained on our dataset)")
    LossCalculator.print_model_loss_for_all_datasets(model, train_loader_p1, test_loader_p1p6, val_loader_p1, device)
    print("--- ---- ---- ---- ---- ---- ")


    # 7. Save the Model
    Util.save_model(model, CHOOSE_MODEL, NUM_EPOCHS)







