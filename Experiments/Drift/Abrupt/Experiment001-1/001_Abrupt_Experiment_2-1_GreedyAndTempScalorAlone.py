import tiktoken
import torch
from datetime import datetime
from DataFineTuning import InputResponseFormatter
from DataLoading import DataLoading
from DataLoading.DataLoading import InstructionDataset
from Datasets import BitextTelecoDS
from LossHandler import LossCalculator
from Models.LLM.GPT2 import GPTModel
from NGramPrefixTree import NGramPrefixTree
from TextGenerationStrategy import Greedy, Odd, TopKTempretureScaling, StrategiesEvaluator
from TextGenerationStrategy import OddMethod1, OddMethod2, OddMethod3, OddMethod4, OddMethod5, OddMethod6, OddMethod7
from Utils import Util
from functools import partial
from datasets import concatenate_datasets
from DriftManager import PlaceholdersUtil, AbruptDriftPlaceholders
import os
from datasets import Dataset, concatenate_datasets

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
    tokenizer.eos_token_id = 50256

    # 4. Load the Dataset - Train, Test, Validation
    train_ratio=0.6
    test_ratio = 0.3
    val_ratio = 0.1
    dataset = BitextTelecoDS.get_bitext_telecom_dataset_splits(train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)
    train_data = dataset["train"]
    test_data = dataset["test"]
    val_data = dataset["validation"]


    # # #todo: remove this on production
    # train_data = train_data.select(range(4))
    # test_data = test_data.select(range(4))
    # val_data = val_data.select(range(4))


    train_data_p1 = BitextTelecoDS.preprocess_bitext_ds2(train_data, "p1", "abrupt")
    val_data_p1 = BitextTelecoDS.preprocess_bitext_ds2(val_data, "p1", "abrupt")



    # Split test into two halves
    n_test = len(test_data)
    half = n_test // 2

    test_data_p1 = BitextTelecoDS.preprocess_bitext_ds2(test_data.select(range(half)), placeholder="p1", drift_type="abrupt")
    test_data_p2 = BitextTelecoDS.preprocess_bitext_ds2(test_data.select(range(half, n_test)), placeholder="p2", drift_type="abrupt")
    # Merge them back



    # Convert your preprocessed lists to Dataset
    test_data_p1 = Dataset.from_list(test_data_p1)
    test_data_p2 = Dataset.from_list(test_data_p2)

    test_data_p1p2 = concatenate_datasets([test_data_p1, test_data_p2])  # preserve their sequence.


    # 5. Data Loaders
    raw_collate = partial(
        InputResponseFormatter.collate_input_targets_with_end_token_and_pad,
        device=device,
        allowed_max_length=1024,
    )

    train_dataset_p1 = InstructionDataset(train_data_p1, tokenizer)
    val_dataset_p1 = InstructionDataset(val_data_p1, tokenizer)
    test_dataset_p1p2 = InstructionDataset(test_data_p1p2, tokenizer)
    train_loader_p1 = DataLoading.make_loader(train_dataset_p1, batch_size=8, shuffle=True, collate_fn=raw_collate)
    val_loader_p1 = DataLoading.make_loader(val_dataset_p1, batch_size=8, shuffle=False, collate_fn=raw_collate)
    test_loader_p1p2 = DataLoading.make_loader(test_dataset_p1p2, batch_size=8, shuffle=False, collate_fn=raw_collate)

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
    # settings, params = gpt_download3.download_and_load_gpt2(
    #     model_size=model_size,
    #     models_dir="gpt2"
    # )
    model = GPTModel.GPTModel(BASE_CONFIG)
    model.to(device)

    # Load your saved fine-tuned checkpoint
    checkpoint_path = "gpt2-medium355M-sft.pth-epochs-10-sft.pth"
    model = Util.load_model(model, checkpoint_path, device)
    model.eval()
    print(f"Loaded fine-tuned model from {checkpoint_path}")


    # 5. Loss Computation on the Pre-trained Model - The loaded one from the saved checkpint checkpoint_path
    print("--- ---- ---- ---- ---- ---- ")
    print("Loss Computation on the Pre-trained Model - The loaded one from the saved checkpint ", checkpoint_path)
    LossCalculator.print_model_loss_for_all_datasets(model, train_loader_p1, test_loader_p1p2, val_loader_p1, device)
    print("--- ---- ---- ---- ---- ---- ")





    # 10. Text Generators
    # 10.1 Greedy
    Greedy.response_strategy_based_extractor_and_saver(model, tokenizer, test_data_p1p2, device, BASE_CONFIG)
    #
    # # 10.2 Topk-TempScaling
    TopKTempretureScaling.response_strategy_based_extractor_and_saver(model, tokenizer, test_data_p1p2, device, BASE_CONFIG)









