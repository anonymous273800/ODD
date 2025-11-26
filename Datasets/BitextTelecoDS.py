import re
from DriftManager import AbruptDriftPlaceholders, IncrementalDriftPlaceholders, GradualDriftPlaceholders




def preprocess_bitext_ds1(dataset, placeholder, drift_type):
   return dataset.map(lambda e: preprocess_bitext_entry1(e, drift_type, placeholder))


def preprocess_bitext_entry1(entry, drift_type, placeholder):
    response = entry["response"]

    # Map drift_type to its corresponding placeholder class
    drift_classes = {
        "abrupt": AbruptDriftPlaceholders,
        "incremental": IncrementalDriftPlaceholders,
        "gradual": GradualDriftPlaceholders,
    }

    if drift_type not in drift_classes:
        raise ValueError(
            f"Invalid drift_type '{drift_type}'. Must be one of {list(drift_classes.keys())}."
        )

    drift_class = drift_classes[drift_type]

    # Dynamically resolve placeholder dict (e.g., PLACEHOLDER_VALUES_P1)
    attr_name = f"PLACEHOLDER_VALUES_{placeholder.upper()}"
    PLACEHOLDER_VALUES = getattr(drift_class, attr_name, {})

    # Replace placeholders in the response text
    for key, value in PLACEHOLDER_VALUES.items():
        response = response.replace(f"{{{{{key}}}}}", value)

    # Normalize and clean response
    response = re.sub(r"\{\{.*?\}\}", "<PLACEHOLDER>", response)
    response = response.replace("\n", " ")
    response = re.sub(r"\s+", " ", response).strip()

    # Update entry
    entry["instruction"] = entry["instruction"].strip()
    entry["response"] = response
    return entry



def preprocess_bitext_ds2(dataset, placeholder, drift_type):
    processed_entries = []

    for i, entry in enumerate(dataset):
        processed_entry = preprocess_bitext_entry2(entry, drift_type, placeholder)
        processed_entries.append(processed_entry)

    return processed_entries





# === Preprocess a single entry (generic for abrupt, incremental, gradual) ===
def preprocess_bitext_entry2(entry, drift_type, placeholder):
    response = entry["response"]

    # Map drift_type to its corresponding placeholder class
    drift_classes = {
        "abrupt": AbruptDriftPlaceholders,
        "incremental": IncrementalDriftPlaceholders,
        "gradual": GradualDriftPlaceholders,
    }

    if drift_type not in drift_classes:
        raise ValueError(f"Invalid drift_type '{drift_type}'. Must be one of {list(drift_classes.keys())}.")

    drift_class = drift_classes[drift_type]

    # Dynamically resolve placeholder dict (e.g., PLACEHOLDER_VALUES_P1)
    attr_name = f"PLACEHOLDER_VALUES_{placeholder.upper()}"
    PLACEHOLDER_VALUES = getattr(drift_class, attr_name, {})

    # Replace placeholders in the response text
    for key, value in PLACEHOLDER_VALUES.items():
        # response = response.replace(f"{{{{{key}}}}}", value)
        response = response.replace(f"{{{{{key}}}}}", f"{{{{{key}}}}} {value}")

    # Normalize and clean response
    # response = re.sub(r"\{\{.*?\}\}", "<PLACEHOLDER>", response)
    response = response.replace("\n", " ")
    response = re.sub(r"\s+", " ", response).strip()

    # Update entry
    entry["instruction"] = entry["instruction"].strip()
    entry["response"] = response
    return entry





'''
https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset
'''
from datasets import load_dataset, DatasetDict, concatenate_datasets

def get_bitext_telecom_dataset_splits(train_ratio=0.7, test_ratio=0.2, val_ratio=0.1, seed=42):

    # Load raw dataset
    raw_dataset = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset")["train"]

    # First split train vs temp (test+val)
    temp_ratio = 1.0 - train_ratio
    train_temp = raw_dataset.train_test_split(test_size=temp_ratio, seed=seed)

    # Split temp into test and validation
    relative_test_ratio = test_ratio / (test_ratio + val_ratio)
    val_test = train_temp["test"].train_test_split(test_size=relative_test_ratio, seed=seed)

    # Build final DatasetDict
    dataset_dict = DatasetDict({
        "train": train_temp["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
        "combined": concatenate_datasets([
            train_temp["train"], val_test["train"], val_test["test"]
        ])
    })

    return dataset_dict





# ===== Main Test =====
if __name__ == "__main__":
    # Example dataset entry
    entry = {
        "instruction": "How can I dispute billing discrepancies?",
        "response": """If you have noticed discrepancies in your bill and wish to contest them, please adhere to the following steps:

        1. Log in to your account on {{WEBSITE_URL}}.
        2. Navigate to the {{INVOICE_SECTION}} section.
        3. Select the bill you wish to dispute.
        4. Click on the {{DISPUTE_INVOICE_OPTION}} to dispute the charge.
        5. Fill in the required information and submit your dispute.

        Our team will review your submission and get back to you within {{DAYS_NUMBER}} business days."""
    }

    processed = preprocess_bitext_entry2(entry, drift_type="abrupt", placeholder="p1")

    print("\n===== ORIGINAL RESPONSE =====")
    print(entry["response"])
    print("\n===== PROCESSED RESPONSE =====")
    print(processed["response"])
