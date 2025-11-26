import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict

def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------- 1. Save per-file results ----------
def save_raw_results(results: Dict[str, pd.DataFrame], out_dir: str, saved_paths: Dict[str, str]):
    for fname, df in results.items():
        raw_csv = Path(out_dir) / f"{Path(fname).stem}_raw.csv"
        df.to_csv(raw_csv, index=False)
        saved_paths[f"{fname}_raw_csv"] = str(raw_csv)


# ---------- 2. Build summary table ----------
def build_summary_table(results: Dict[str, pd.DataFrame], out_dir: str, saved_paths: Dict[str, str]) -> pd.DataFrame:
    summary_rows = []
    for fname, df in results.items():
        row = {"file": fname}
        for col in df.columns:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std()
            row[f"{col}_median"] = df[col].median()
            row[f"{col}_min"] = df[col].min()
            row[f"{col}_max"] = df[col].max()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("file")

    summary_csv = Path(out_dir) / "summary.csv"
    summary_tex = Path(out_dir) / "summary.tex"
    summary_df.round(3).to_csv(summary_csv)
    with open(summary_tex, "w") as f:
        f.write(summary_df.round(3).to_latex(escape=False))

    saved_paths["summary_csv"] = str(summary_csv)
    saved_paths["summary_tex"] = str(summary_tex)
    return summary_df


# ---------- 3. Plot helper to prepare melted dataframe ----------
def prepare_plot_df(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert results into a melted DataFrame suitable for Seaborn plotting.
    Normalizes ChrF scores and applies canonical DISPLAY labels only.
    """
    melted = []
    for fname, df in results.items():
        for metric in df.columns:
            for val in df[metric].values:
                melted.append({"file": fname, "metric": metric, "score": val})
    plot_df = pd.DataFrame(melted)

    # Normalize ChrF to [0,1]
    if "ChrF" in plot_df["metric"].unique():
        plot_df.loc[plot_df["metric"] == "ChrF", "score"] /= 100.0

    # Canonical *display* labels (leave them verbose for the full plots)
    canonical = {
        # LLM-Greedy
        "LLM-Greedy": "LLM-Greedy",
        "Greedy": "LLM-Greedy",
        "response_greedy.json": "LLM-Greedy",

        # LLM-Temp Scaled
        "LLM-TempScaled": "LLM-Temp Scaled",
        "LLM-Temp Scaled": "LLM-Temp Scaled",
        "response_topk_tempscaling.json": "LLM-Temp Scaled",

        # ODD (various aliases collapse to the requested label)
        "ODD": "ODD(LLM + Prefix Tree)",
        "SPINE": "ODD(LLM + Prefix Tree)",
        "SPINE (LLM + Prefix-Tree)": "ODD(LLM + Prefix Tree)",
        "SPINE (LLM + Prefix Tree)": "ODD(LLM + Prefix Tree)",
        "ODD (LLM + Prefix-Tree)": "ODD(LLM + Prefix Tree)",
        "ODD (LLM + Prefix Tree)": "ODD(LLM + Prefix Tree)",
        "response_odd.json": "ODD(LLM + Prefix Tree)",
        "ODD(LLM + Prefix Tree)": "ODD(LLM + Prefix Tree)",
    }
    plot_df["file"] = plot_df["file"].map(lambda x: canonical.get(x, x))

    return plot_df



# ---------- 6. Focused Barplot: LLM vs ODD ----------
def plot_barplot_llm_vs_odd(plot_df: pd.DataFrame, out_dir: str, saved_paths: Dict[str, str]):
    subset = plot_df[plot_df["file"].isin(["LLM-Greedy", "ODD(LLM + Prefix Tree)"])].copy()
    subset["legend_label"] = subset["file"].replace({
        "LLM-Greedy": "LLM",
        "ODD(LLM + Prefix Tree)": "ODD",
    })

    uniq = subset["legend_label"].unique().tolist()
    if len(uniq) == 0:
        print("[Warning] No LLM-Greedy / ODD data found.")
        return
    elif len(uniq) == 1:
        print(f"[Warning] Only one model present ({uniq[0]}). Plot will show single bars.")

    base_palette = {"LLM": "#9ca3af", "ODD": "#2ca02c"}
    palette = [base_palette[u] for u in uniq]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=subset,
        x="metric",
        y="score",
        hue="legend_label",
        errorbar=("ci", 95),
        capsize=0.1,
        palette=palette
    )

    plt.title("LLM vs. ODD: NLG Metrics (Mean ± 95% CI)", fontsize=13)
    plt.xlabel("Metric", fontsize=12)
    plt.ylabel("Mean Score", fontsize=12)
    plt.xticks(rotation=0)

    # --- Legend above the plot (centered, no title) ---
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles, labels,
        title=None,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave space at top

    out_path = Path(out_dir) / "nlg_barplot_llm_vs_odd.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths["barplot_llm_vs_odd"] = str(out_path)


# ---------- 4. Boxplot for all models ----------
def plot_boxplot_all(plot_df: pd.DataFrame, out_dir: str, saved_paths: Dict[str, str]):
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", n_colors=len(plot_df["file"].unique()))

    sns.boxplot(
        data=plot_df,
        x="metric",
        y="score",
        hue="file",
        showmeans=True,
        palette=palette,
        flierprops=dict(marker='o', markersize=4, alpha=0.9, markeredgewidth=0.6)
    )
    plt.title("NLG Metric Distributions by Model", fontsize=13)
    plt.xlabel("Metric", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend(title="Model", loc="best", frameon=True)
    plt.xticks(rotation=20)
    plt.tight_layout()

    out_path = Path(out_dir) / "nlg_boxplot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths["boxplot"] = str(out_path)


# ---------- 5. Barplot for all models ----------
def plot_barplot_all(plot_df: pd.DataFrame, out_dir: str, saved_paths: Dict[str, str]):
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", n_colors=len(plot_df["file"].unique()))

    sns.barplot(
        data=plot_df,
        x="metric",
        y="score",
        hue="file",
        errorbar=("ci", 95),
        capsize=0.1,
        palette=palette
    )

    plt.title("NLG Metrics (Mean ± 95% CI)", fontsize=13)
    plt.xlabel("Metric", fontsize=12)
    plt.ylabel("Mean Score", fontsize=12)
    plt.xticks(rotation=0)

    # --- Legend above the plot (centered) ---
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles, labels,
        title=None,                # remove title
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),          # spread horizontally
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave space at top for legend

    out_path = Path(out_dir) / "nlg_barplot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths["barplot_all"] = str(out_path)



# ---------- 6. Focused Barplot: LLM vs SPINE ----------
def plot_barplot_llm_vs_spine(plot_df: pd.DataFrame, out_dir: str, saved_paths: Dict[str, str]):
    # Filter for only LLM and SPINE
    subset = plot_df[plot_df["file"].isin(["LLM", "SPINE"])].copy()

    plt.figure(figsize=(10, 6))
    palette = ["#9ca3af", "#2ca02c"]  # gray for LLM, green for SPINE

    sns.barplot(
        data=subset,
        x="metric",
        y="score",
        hue="file",
        errorbar=("ci", 95),
        capsize=0.1,
        palette=palette
    )

    plt.title("LLM vs. SPINE: NLG Metrics (Mean ± 95% CI)", fontsize=13)
    plt.xlabel("Metric", fontsize=12)
    plt.ylabel("Mean Score", fontsize=12)
    plt.legend(title="Model", loc="best", frameon=True)
    plt.xticks(rotation=0)
    plt.tight_layout()

    out_path = Path(out_dir) / "nlg_barplot_llm_vs_spine.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths["barplot_llm_vs_spine"] = str(out_path)


# ---------- 7. Main orchestrator ----------
def save_nlg_results(results: Dict[str, pd.DataFrame], out_dir: str) -> Dict[str, str]:
    _ensure_dir(out_dir)
    saved_paths = {}

    save_raw_results(results, out_dir, saved_paths)
    summary_df = build_summary_table(results, out_dir, saved_paths)
    plot_df = prepare_plot_df(results)

    plot_boxplot_all(plot_df, out_dir, saved_paths)
    plot_barplot_all(plot_df, out_dir, saved_paths)
    plot_barplot_llm_vs_spine(plot_df, out_dir, saved_paths)

    return saved_paths
