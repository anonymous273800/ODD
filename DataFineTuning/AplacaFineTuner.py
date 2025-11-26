from Datasets import BitextTelecoDS


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text #+ input_text


if __name__ == "__main__":
    dataset = BitextTelecoDS.get_bitext_telecom_dataset_splits()
    dataset_train = dataset["train"]
    print(dataset_train[0])
    instruction = dataset_train[0]
    response = dataset_train[0]["response"]
    finetuned_instruction = format_input(instruction)
    print("finetuned_instruction: ",finetuned_instruction)
    print('-------------------')
    desired_response = f"\n\n### Response:\n{response}"
    print("desired_response: ",desired_response)
    print("")
    print("Final Entry: ")
    print(finetuned_instruction + desired_response)