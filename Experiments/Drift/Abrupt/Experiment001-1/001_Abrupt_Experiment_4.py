import os
import torch
import numpy as np
from Evaluation import OllamaEvaluator, OllamaScoresPresenter
from Utils import Constants, Util

if __name__ == "__main__":
    # 1. set seed
    torch.manual_seed(42)

    EXPERIMENT_PATH = os.path.dirname(os.path.abspath(__file__))
# --------------------------------------------------------
    # 2. Ollama Evaluation
    # --------------------------------------------------------
    ollama_running = OllamaEvaluator.check_if_running("ollama")
    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", ollama_running)



    # Greedy
    print("\n[Ollama] Greedy Evaluation")
    test_data = Util.load_json_file(Constants.RESPONSE_GREEDY_FILE_NAME, base_path=EXPERIMENT_PATH)
    print("test_data", test_data)
    scores_greedy = OllamaEvaluator.generate_model_scores(test_data, "model_response", model="llama3")
    print("Scores:", scores_greedy)
    print("Mean:", np.mean(scores_greedy))

    # Top-k Temp Scaling
    print("\n[Ollama] Top-k Temp Scaling Evaluation")
    test_data = Util.load_json_file(Constants.RESPONSE_TOPK_TEMPSCALING_FILE_NAME, base_path=EXPERIMENT_PATH)
    scores_topk = OllamaEvaluator.generate_model_scores(test_data, "model_response", model="llama3")
    print("Scores:", scores_topk)
    print("Mean:", np.mean(scores_topk))

    # SPINE
    print("\n[Ollama] SPINE Evaluation")
    test_data = Util.load_json_file(Constants.RESPONSE_SPINE_FILE_NAME, base_path=EXPERIMENT_PATH)
    scores_spine = OllamaEvaluator.generate_model_scores(test_data, "model_response", model="llama3")
    print("Scores:", scores_spine)
    print("Mean:", np.mean(scores_spine))

    # Comparison + plots
    out_dir1 = os.path.join(EXPERIMENT_PATH, "evaluation_outputs_ollama")
    OllamaScoresPresenter.compare_and_plot(scores_greedy, scores_topk, scores_spine, out_dir1)