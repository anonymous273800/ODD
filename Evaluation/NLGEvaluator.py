import os
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import Levenshtein
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from pathlib import Path
import seaborn as sns
from sacrebleu.metrics import CHRF, BLEU, TER
from math import pi
from Evaluation import NLGEvaluationSaver
chrf = CHRF()

# Load embedding model once (global to avoid reloading every time)
_sentence_model = None # SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# ---------- Metric functions ----------
def exact_match(reference: str, prediction: str) -> int:
    return int(reference.strip() == prediction.strip())


# def edit_distance(reference: str, prediction: str) -> int:
#     return Levenshtein.distance(reference, prediction)

# Why normalized -> Raw Levenshtein grows with text length, so two longer texts always look "worse."
# Normalized keeps results comparable across different response lengths.
def normalized_edit_distance(reference, prediction):
    dist = Levenshtein.distance(reference, prediction)
    return 1 - dist / max(len(reference), len(prediction), 1)

def chrf_score(reference: str, prediction: str) -> float:
    return chrf.sentence_score(prediction, [reference]).score

def bleu_score(reference: str, prediction: str) -> float:
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], prediction.split(), smoothing_function=smoothie)


def rouge_l(reference: str, prediction: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, prediction)['rougeL'].fmeasure


# def cosine_similarity1(reference: str, prediction: str) -> float:
#     emb1 = _sentence_model.encode(reference, convert_to_tensor=True)
#     emb2 = _sentence_model.encode(prediction, convert_to_tensor=True)
#     return float(util.pytorch_cos_sim(emb1, emb2).item())


def cosine_similarity(reference: str, prediction: str) -> float:
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer, util
        _sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    else:
        from sentence_transformers import util   # <-- make sure util is always in scope

    emb1 = _sentence_model.encode(reference, convert_to_tensor=True)
    emb2 = _sentence_model.encode(prediction, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2).item())



def bert_score_metric(references: List[str], predictions: List[str]) -> List[float]:
    """Compute BERTScore F1 for a batch of predictions."""
    _, _, F1 = bert_score(
        predictions, references,
        lang="en", model_type="bert-base-uncased",
        verbose=False
    )
    return F1.tolist()


# ---------- Core evaluation ----------
# def evaluate_nlg_file(file_path: str) -> pd.DataFrame:
#     """Evaluate all metrics for one JSON file and return a DataFrame of per-sample results."""
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#
#     rows, refs, preds = [], [], []
#
#     for entry in tqdm(data, desc=f"Evaluating {os.path.basename(file_path)}"):
#         reference = entry["response"]
#         prediction = entry["model_response"]
#
#         rows.append({
#             "exact_match": exact_match(reference, prediction),
#             "edit_distance": normalized_edit_distance(reference, prediction),
#             "bleu": bleu_score(reference, prediction),
#             "rougeL": rouge_l(reference, prediction),
#             "cosine_sim": cosine_similarity(reference, prediction),
#             # bert_score added later
#         })
#
#         refs.append(reference)
#         preds.append(prediction)
#
#     # Add BERTScore in batch
#     bert_scores = bert_score_metric(refs, preds)
#     for i, row in enumerate(rows):
#         row["bert_score"] = bert_scores[i]
#
#     return pd.DataFrame(rows)


# --- Label mappings (global) ---
MODEL_LABELS = {
    "response_greedy.json": "LLM-Greedy",
    "response_topk_tempscaling.json": "LLM-Temp Scaled",
    "response_odd.json": "ODD(LLM + Prefix Tree)",   # <- exact label requested
}



METRIC_LABELS = {
    "exact_match": "Exact Match",
    "edit_distance": "Edit Distance",
    "bleu": "BLEU",
    "rougeL": "ROUGE-L",
    "cosine_sim": "Cosine Similarity",
    "chrf": "ChrF",
    "bert_score": "BERTScore",
}


def evaluate_nlg_file(file_path: str, out_dir: str = None) -> pd.DataFrame:
    """
    Evaluate all metrics for one JSON file and return a DataFrame of per-sample results.
    Optionally saves correlation heatmap if out_dir is provided.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows, refs, preds = [], [], []

    for entry in tqdm(data, desc=f"Evaluating {os.path.basename(file_path)}"):
        reference = entry["response"]
        prediction = entry["model_response"]

        rows.append({
            "exact_match": exact_match(reference, prediction),
            "edit_distance": normalized_edit_distance(reference, prediction),
            "bleu": bleu_score(reference, prediction),
            "rougeL": rouge_l(reference, prediction),
            "cosine_sim": cosine_similarity(reference, prediction),
            "chrf": chrf_score(reference, prediction),   # NEW
            # bert_score added later
        })

        refs.append(reference)
        preds.append(prediction)

    # Add BERTScore in batch
    bert_scores = bert_score_metric(refs, preds)
    for i, row in enumerate(rows):
        row["bert_score"] = bert_scores[i]

    df = pd.DataFrame(rows)
    df.rename(columns=METRIC_LABELS, inplace=True)

    # Optionally save correlation heatmap
    if out_dir is not None:
        _ensure_dir(out_dir)
        plot_correlation_heatmap(
            df,
            out_dir,
            f"{MODEL_LABELS.get(Path(file_path).name, Path(file_path).stem)}_correlation.png"
        )

    return df





def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# def save_nlg_results(results: Dict[str, pd.DataFrame], out_dir: str) -> Dict[str, str]:
#     """
#     Save NLG evaluation results (per-file, summary, plots).
#
#     Parameters
#     ----------
#     results : Dict[str, pd.DataFrame]
#         Mapping from file name -> DataFrame of per-sample metrics.
#     out_dir : str
#         Directory to save outputs.
#
#     Returns
#     -------
#     Dict[str, str]
#         Dictionary of saved file paths.
#     """
#     _ensure_dir(out_dir)
#     saved_paths = {}
#
#     # ---------- 1. Save per-file raw results ----------
#     for fname, df in results.items():
#         raw_csv = Path(out_dir) / f"{Path(fname).stem}_raw.csv"
#         df.to_csv(raw_csv, index=False)
#         saved_paths[f"{fname}_raw_csv"] = str(raw_csv)
#
#     # ---------- 2. Build summary table ----------
#     summary_rows = []
#     for fname, df in results.items():
#         row = {"file": fname}
#         for col in df.columns:
#             row[f"{col}_mean"] = df[col].mean()
#             row[f"{col}_std"] = df[col].std()
#             row[f"{col}_median"] = df[col].median()
#             row[f"{col}_min"] = df[col].min()
#             row[f"{col}_max"] = df[col].max()
#         summary_rows.append(row)
#
#     summary_df = pd.DataFrame(summary_rows).set_index("file")
#
#     summary_csv = Path(out_dir) / "summary.csv"
#     summary_tex = Path(out_dir) / "summary.tex"
#     summary_df.round(3).to_csv(summary_csv)
#     with open(summary_tex, "w") as f:
#         f.write(summary_df.round(3).to_latex(escape=False))
#
#     saved_paths["summary_csv"] = str(summary_csv)
#     saved_paths["summary_tex"] = str(summary_tex)
#
#     # ---------- 3. Plots (publication-ready) ----------
#     melted = []
#     for fname, df in results.items():
#         for metric in df.columns:
#             for val in df[metric].values:
#                 melted.append({"file": fname, "metric": metric, "score": val})
#     plot_df = pd.DataFrame(melted)
#
#     # --- Normalize ChrF to [0,1] for consistent scaling ---
#     if "ChrF" in plot_df["metric"].unique():
#         plot_df.loc[plot_df["metric"] == "ChrF", "score"] /= 100.0
#
#     # Boxplot
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(data=plot_df, x="metric", y="score", hue="file", showmeans=True)
#     plt.title("NLG Metric Distributions by File")
#     plt.xticks(rotation=20)
#     plt.tight_layout()
#     boxplot_path = Path(out_dir) / "nlg_boxplot.png"
#     plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     saved_paths["boxplot"] = str(boxplot_path)
#
#     # Bar plot (mean ± std)
#     plt.figure(figsize=(10, 6))
#     sns.barplot(
#         data=plot_df, x="metric", y="score", hue="file",
#         errorbar=("ci", 95), capsize=0.1
#     )
#     plt.legend(title=None)
#     plt.title("NLG Metrics (Mean ± 95% CI)")
#     plt.xticks(rotation=0)
#     plt.tight_layout()
#     barplot_path = Path(out_dir) / "nlg_barplot.png"
#     plt.savefig(barplot_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     saved_paths["barplot"] = str(barplot_path)
#
#     return saved_paths


# ---------- 7. Main orchestrator ----------
# ---------- 7. Main orchestrator ----------
# ---------- 7. Main orchestrator ----------
def save_nlg_results(results: Dict[str, pd.DataFrame], out_dir: str) -> Dict[str, str]:
    _ensure_dir(out_dir)
    saved_paths = {}

    NLGEvaluationSaver.save_raw_results(results, out_dir, saved_paths)
    NLGEvaluationSaver.build_summary_table(results, out_dir, saved_paths)
    plot_df = NLGEvaluationSaver.prepare_plot_df(results)

    NLGEvaluationSaver.plot_boxplot_all(plot_df, out_dir, saved_paths)
    NLGEvaluationSaver.plot_barplot_all(plot_df, out_dir, saved_paths)
    NLGEvaluationSaver.plot_barplot_llm_vs_odd(plot_df, out_dir, saved_paths)  # <- renamed

    return saved_paths






def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str, filename: str = "correlation_heatmap.png") -> str:
    """
    Generate and save a correlation heatmap for evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing metric columns to correlate.
    out_dir : str
        Directory to save the heatmap.
    filename : str, optional
        Name of the output file (default: correlation_heatmap.png).

    Returns
    -------
    str
        Path to the saved heatmap image.
    """
    out_path = Path(out_dir) / filename

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Metric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return str(out_path)




def radar_chart(summary_df, out_path):
    metrics = [c for c in summary_df.columns if c.endswith("_mean")]
    labels = list(summary_df.index)
    N = len(metrics)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)

    for label in labels:
        values = summary_df.loc[label, metrics].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.savefig(out_path, dpi=300)
    plt.close()


def evaluate_nlg_files(files: List[str], base_path: str, sub_path: str) -> Dict[str, pd.DataFrame]:
    """
    Evaluate multiple JSON files containing model outputs.

    Parameters
    ----------
    files : List[str]
        List of JSON file names.
    base_path : str
        Path where files are stored.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping file name -> DataFrame of per-sample results.
    """
    results = {}
    for file in files:
        file_path = os.path.join(base_path, sub_path, file)
        df = evaluate_nlg_file(file_path)

        # --- Apply human-readable model name ---
        display_name = MODEL_LABELS.get(file, file)
        results[display_name] = df
    return results


