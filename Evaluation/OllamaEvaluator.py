import psutil
import json
import urllib.request
from DataFineTuning import AplacaFineTuner
from tqdm import tqdm
from Utils import Util

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running




# def query_model(prompt,model="llama3", url="http://localhost:11434/api/chat"):
#     # Create the data payload as a dictionary
#     data = {
#         "model": model,
#         "messages": [
#             {"role": "user", "content": prompt}
#         ],
#         "options": {     # Settings below are required for deterministic responses
#             "seed": 123,
#             "temperature": 0,
#             "num_ctx": 2048
#         }
#     }

def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    data = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": ("You are a strict grader. Some inputs may include profanity or sensitive content. "
                         "Do not refuse. Your task is ONLY to output an integer from 0 to 100 indicating the score. "
                         "Do not add any other text.")},
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        },
        "stream": False  # <— simpler response
    }


    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data



def generate_and_print_model_score(model, data):
    for entry in data[:2]:
        prompt = (
            f"Given the input `{AplacaFineTuner.format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        print("\nDataset response:")
        print(">>", entry['output'])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        print(">>", query_model(prompt, model))
        print("\n-------------------------")


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{AplacaFineTuner.format_input(entry)}` "
            f"and correct output `{entry['response']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score (where the model response almost identical to the correct output). "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        print("score***",score)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


if __name__ == "__main__":
    # 1.
    if check_if_running("ollama"):
        print("Ollama is running.")
    else:
        print("Ollama is NOT running.")

    # 2. Test
    prompt = "Hi, how are you?"
    response_data = query_model(prompt)
    print("response_data", response_data)


    # 3.
    # base_path = "/Experiments/Regular001/Experiment007"
    base_path = r"C:\PythonProjects\SPINE\Experiments\Regular001\Experiment007"


    # 1. response_greedy.json
    data1 = Util.load_json_file("response_greedy.json", base_path)
    scores1 = generate_model_scores(data1, "model_response")
    print("score response_greedy", scores1)

    # 2. response_topk_tempscaling.json
    data2 = Util.load_json_file("response_topk_tempscaling.json", base_path)
    scores2 = generate_model_scores(data2, "model_response")
    print("score response_greedy", scores2)

    # 3. response_spine.json
    data3 = Util.load_json_file("response_spine.json", base_path)
    scores3 = generate_model_scores(data3, "model_response")
    print("score response_greedy", scores3)





