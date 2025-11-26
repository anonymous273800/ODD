from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel


# ---------- Global plotting style ----------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------- Utilities ----------
def _as_clean_array(scores: List[float]) -> np.ndarray:
    """Convert to 1D float array and drop NaN values."""
    arr = np.asarray(scores, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def _ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------- Statistics ----------
def summarize(scores: List[float]) -> Dict[str, float]:
    """Compute descriptive statistics for a single list of scores."""
    x = _as_clean_array(scores)
    if x.size == 0:
        return dict(mean=np.nan, std=np.nan, median=np.nan,
                    q1=np.nan, q3=np.nan, iqr=np.nan,
                    min=np.nan, max=np.nan, n=0)

    q1, q3 = np.percentile(x, [25, 75])
    return dict(
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        median=float(np.median(x)),
        q1=float(q1),
        q3=float(q3),
        iqr=float(q3 - q1),
        min=float(np.min(x)),
        max=float(np.max(x)),
        n=int(x.size),
    )


def summarize_many(score_dict: Dict[str, List[float]]) -> pd.DataFrame:
    """Summarize multiple methods into one DataFrame."""
    rows = []
    for name, scores in score_dict.items():
        stats = summarize(scores)
        stats["method"] = name
        rows.append(stats)

    df = pd.DataFrame(rows).set_index("method")
    cols = ["mean", "std", "median", "q1", "q3", "iqr", "min", "max", "n"]
    return df[cols]


def paired_tests(score_dict: Dict[str, List[float]]) -> pd.DataFrame:
    """Perform paired t-tests between all method pairs."""
    names = list(score_dict.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a = _as_clean_array(score_dict[a_name])
            b = _as_clean_array(score_dict[b_name])

            n = min(a.size, b.size)
            if n < 2:
                t_stat, p_val = np.nan, np.nan
            else:
                t_stat, p_val = ttest_rel(a[:n], b[:n])

            rows.append(dict(
                method_a=a_name,
                method_b=b_name,
                n=n,
                t_stat=float(t_stat) if np.isfinite(t_stat) else np.nan,
                p_value=float(p_val) if np.isfinite(p_val) else np.nan
            ))
    return pd.DataFrame(rows)


# ---------- Save helpers ----------
def save_summary_csv(df: pd.DataFrame, out_dir: str, filename: str = "summary.csv") -> str:
    _ensure_dir(out_dir)
    path = str(Path(out_dir) / filename)
    df.to_csv(path, index=True)
    return path


def save_latex_table(df: pd.DataFrame, out_dir: str, filename: str = "summary.tex") -> str:
    """Save DataFrame as LaTeX table for papers."""
    _ensure_dir(out_dir)
    path = str(Path(out_dir) / filename)
    with open(path, "w") as f:
        f.write(df.round(3).to_latex(index=True, escape=False))
    return path


# ---------- Plotting ----------
def plot_bar_with_ci(score_dict: Dict[str, List[float]], out_dir: str,
                     filename: str = "bar_means_ci.png",
                     title: str = "Mean Scores (95% CI)") -> str:
    """Generate and save a bar chart with bootstrapped 95% CI. Auto-adjust width."""
    _ensure_dir(out_dir)
    df = pd.DataFrame([
        {"method": m, "score": s}
        for m, scores in score_dict.items()
        for s in _as_clean_array(scores)
    ])

    n_methods = df["method"].nunique()
    fig_width = max(5, n_methods * 1.5)  # dynamic width
    plt.figure(figsize=(fig_width, 6))

    ax = sns.barplot(
        data=df, x="method", y="score",
        hue="method", legend=False,
        palette="Set2", errorbar=("ci", 95),
        capsize=0.1, width=0.6
    )
    ax.set_ylabel("Mean Score")
    ax.set_title(title)
    plt.xticks(rotation=15)

    out_path = str(Path(out_dir) / filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    return out_path


def plot_boxplot(score_dict: Dict[str, List[float]], out_dir: str,
                 filename: str = "boxplot_scores.png",
                 title: str = "Scores by Method") -> str:
    """Generate and save a publication-quality boxplot with strip points."""
    _ensure_dir(out_dir)
    df = pd.DataFrame([
        {"method": m, "score": s}
        for m, scores in score_dict.items()
        for s in _as_clean_array(scores)
    ])

    n_methods = df["method"].nunique()
    fig_width = max(5, n_methods * 1.5)
    plt.figure(figsize=(fig_width, 6))

    ax = sns.boxplot(
        data=df, x="method", y="score",
        hue="method", legend=False,
        showmeans=True, palette="Set2"
    )
    sns.stripplot(data=df, x="method", y="score", ax=ax, color=".25", size=4)
    ax.set_ylabel("Score")
    ax.set_title(title)
    plt.xticks(rotation=15)

    out_path = str(Path(out_dir) / filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    return out_path


# ---------- All-in-one helper ----------
def compare_and_plot(scores_greedy: List[float],
                     scores_topk_temp: List[float],
                     scores_spine: List[float],
                     out_dir: str = "evaluation_outputs"
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    One-shot evaluation pipeline for three scoring methods.
    Signature is fixed for paper reproducibility.
    """
    score_dict = {
        "Greedy": scores_greedy,
        "Top-k Temp Scaling": scores_topk_temp,
        "ODD": scores_spine,
    }

    summary_df = summarize_many(score_dict)
    tests_df = paired_tests(score_dict)

    _ensure_dir(out_dir)
    paths = {
        "summary_csv": save_summary_csv(summary_df, out_dir),
        "summary_tex": save_latex_table(summary_df, out_dir),
        "tests_csv": save_summary_csv(tests_df, out_dir, "paired_tests.csv"),
        "tests_tex": save_latex_table(tests_df, out_dir, "paired_tests.tex"),
        "boxplot": plot_boxplot(score_dict, out_dir),
        "barplot": plot_bar_with_ci(score_dict, out_dir),
    }
    return summary_df, tests_df, paths


# ---------- Example main ----------
if __name__ == "__main__":
    # Example scores (replace with your experimental results)
    scores_greedy = [75, 81, 79, 84, 90, 88, 83]
    scores_topk_temp = [73, 80, 78, 82, 88, 87, 82]
    scores_spine = [80, 86, 85, 90, 92, 91, 89]

    BASE_PATH = Path(__file__).resolve().parent
    out_dir = BASE_PATH / "evaluation_outputs"

    summary, tests, paths = compare_and_plot(
        scores_greedy, scores_topk_temp, scores_spine, out_dir=out_dir
    )

    print("\nSummary statistics:\n", summary.round(2))
    print("\nPaired t-tests:\n", tests.round(4))
    print("\nSaved files:")
    for key, val in paths.items():
        print(f"  - {key}: {val}")
