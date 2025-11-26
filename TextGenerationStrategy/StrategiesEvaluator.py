import Utils.Constants
from Utils import Util, Constants
from Evaluation import OllamaEvaluator, OllamaScoresPresenter
import os
import numpy as np
from Evaluation import NLGEvaluator

def evaluate(base_path):
    # --------------------------------------------------------
    # 2. Natural Language Generation (NLG) evaluation
    # --------------------------------------------------------

    print("\nRunning NLG evaluation metrics (BLEU, ROUGE-L, BERTScore, etc.)...")
    nlg_results = NLGEvaluator.evaluate_nlg_files(
        files=[
            Constants.RESPONSE_GREEDY_FILE_NAME,
            Constants.RESPONSE_TOPK_TEMPSCALING_FILE_NAME,
            Constants.RESPONSE_SPINE_FILE_NAME
        ],
        base_path=base_path,
        sub_path=Utils.Constants.PREDICTIONS_PATH
    )

    # Save + present NLG results
    out_dir2 = os.path.join(base_path, "evaluation_outputs_ngl")
    NLGEvaluator.save_nlg_results(nlg_results, out_dir2)

    print("\nEvaluation complete. Results saved in:", out_dir2)

    # # --------------------------------------------------------
    # # 2. Ollama Evaluation
    # # --------------------------------------------------------
    # ollama_running = OllamaEvaluator.check_if_running("ollama")
    # if not ollama_running:
    #     raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    # print("Ollama running:", ollama_running)
    #
    #
    #
    # # Greedy
    # print("\n[Ollama] Greedy Evaluation")
    # test_data = Util.load_json_file(Constants.RESPONSE_GREEDY_FILE_NAME, base_path=base_path)
    # scores_greedy = OllamaEvaluator.generate_model_scores(test_data, "model_response", model="llama3")
    # print("Scores:", scores_greedy)
    # print("Mean:", np.mean(scores_greedy))
    #
    # # Top-k Temp Scaling
    # print("\n[Ollama] Top-k Temp Scaling Evaluation")
    # test_data = Util.load_json_file(Constants.RESPONSE_TOPK_TEMPSCALING_FILE_NAME, base_path=base_path)
    # scores_topk = OllamaEvaluator.generate_model_scores(test_data, "model_response", model="llama3")
    # print("Scores:", scores_topk)
    # print("Mean:", np.mean(scores_topk))
    #
    # # SPINE
    # print("\n[Ollama] SPINE Evaluation")
    # test_data = Util.load_json_file(Constants.RESPONSE_SPINE_FILE_NAME, base_path=base_path)
    # scores_spine = OllamaEvaluator.generate_model_scores(test_data, "model_response", model="llama3")
    # print("Scores:", scores_spine)
    # print("Mean:", np.mean(scores_spine))
    #
    # # Comparison + plots
    # out_dir1 = os.path.join(base_path, "evaluation_outputs_ollama")
    # OllamaScoresPresenter.compare_and_plot(scores_greedy, scores_topk, scores_spine, out_dir1)




if __name__ == "__main__":
    evaluate(base_path="C:\PythonProjects\ODD\Experiments\Drift\Abrupt\Experiment001")
