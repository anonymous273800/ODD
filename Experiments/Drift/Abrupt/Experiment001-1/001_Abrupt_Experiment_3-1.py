import os

import torch

from TextGenerationStrategy import StrategiesEvaluator

if __name__ == "__main__":

    # 1. set seed
    torch.manual_seed(42)






    # 11. Evaluations
    EXPERIMENT_PATH = os.path.dirname(os.path.abspath(__file__))
    StrategiesEvaluator.evaluate(base_path=EXPERIMENT_PATH)














