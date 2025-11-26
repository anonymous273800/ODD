import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ---------- Metric functions ----------
def exact_match(reference: str, prediction: str) -> int:
    return int(reference.strip() == prediction.strip())

def normalized_edit_distance(reference, prediction):
    dist = Levenshtein.distance(reference, prediction)
    print("--- dist",dist)
    return 1 - dist / max(len(reference), len(prediction), 1)


def bleu_score(reference: str, prediction: str) -> float:
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], prediction.split(), smoothing_function=smoothie)

reference = "Mohammad Ahmad abdelmajeed Abu-Shaira"
prediction = "Mohammad Ahmad abdelmajeed Abu-Shaira"
exact_match_result = exact_match(reference, prediction)
print("exact_match_result: ", exact_match_result)
print("-------------------------")

norm_edit_distance = normalized_edit_distance(reference, prediction)
print("norm_edit_distance: ", norm_edit_distance)

print("-------------------------")

bleu_score_result = bleu_score(reference, prediction)
print("bleu_score_result: ", bleu_score_result)