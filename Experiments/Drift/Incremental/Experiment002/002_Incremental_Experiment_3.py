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




    # 11. Evaluations
    EXPERIMENT_PATH = os.path.dirname(os.path.abspath(__file__))
    StrategiesEvaluator.evaluate(base_path=EXPERIMENT_PATH)














