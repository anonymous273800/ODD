from tqdm import tqdm
import math
import Utils.Constants
from DataFineTuning import AplacaFineTuner
from Utils import Util
import torch
import torch.nn.functional as F
import numpy as np
from TextGenerationStrategy import OddMethod1, OddMethod2, OddMethod3, OddMethod4, OddMethod5, OddMethod6,OddMethod7
torch.set_printoptions(threshold=torch.inf)




def compute_gamma(llm_c, trie_c):
    denom = llm_c + trie_c
    return (llm_c / denom) if denom > 0 else 0.5



def response_strategy_based_extractor_and_saver_abrupt1(model, tokenizer, test_data, device, BASE_CONFIG, trie1,
                                                       trie2, SCORING_CONFIG):
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
        print("ODD Method1 - No C")
        token_ids1 = OddMethod1.generate_greedy1(
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
        print("Prompt length:", prompt_len, " Generated length1:", token_ids1.size(1), " Continuation length1:", token_ids1.size(1) - prompt_len)

        # Decode only continuation
        continuation_tokenstoken_ids1 = token_ids1[:, prompt_len:]
        response_text1 = Util.token_ids_to_text(continuation_tokenstoken_ids1, tokenizer)


        # Clean up special markers
        response_text1 = response_text1.replace("### Response:", "").replace("<|endoftext|>", "").strip()
        print("model response1=>", response_text1)

        test_data[i]["model_response_1"] = response_text1
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ END ODD_1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")



        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("response=>", test_data[i]["response"])
    Util.store_in_json_file(test_data, Utils.Constants.RESPONSE_SPINE_FILE_NAME1)


if __name__ == "__main__":
    trie_scores = [0.5, 0.8, 0.6]
    trie_scores_tensor = torch.tensor(trie_scores)
    probab = Util.normalize_trie_scores(trie_scores_tensor)
    print(probab)
