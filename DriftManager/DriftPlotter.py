import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_drift_incremental(results_list, title="Incremental Drift Progression (P1→P5)", save_path=None, size="normal"):
    """
    Plot incremental drift across steps (e.g., P1→P2, P2→P3, ..., P1→P5).

    Parameters
    ----------
    results_list : list[tuple[str, dict]]
        Each item is (step_label, results_dict). results_dict may include:
            - "avg_lexical_drift"        in [0,1]  (higher = more different)
            - "avg_semantic_similarity"  in [0,1]  (higher = more similar)
            - "bertscore_f1"             in [0,1]  (higher = more similar)  [optional]
            - "mmd_rbf"                  unbounded; will be normalized to [0,1] as mmd/(1+mmd)  [optional]
    title : str
        Figure title.
    save_path : str | None
        If provided, saves the figure (PDF, 300 dpi).
    size : {"normal","small"}
        Controls figure size and font scaling.
    """

    # --- Style / Aesthetics ---
    plt.style.use('seaborn-v0_8-paper')
    colors = sns.color_palette("colorblind", 4)  # will use up to 4 metrics

    if size == "small":
        figsize = (6.5, 3.6)
        font_cfg = {"font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8}
        lw = 1.5
        ms = 6
    else:
        figsize = (9, 5)
        font_cfg = {"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10}
        lw = 2.0
        ms = 8

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], **font_cfg})

    # --- Helpers ---
    def _norm_mmd(val):
        if val is None:
            return None
        val = float(val)
        return val / (1.0 + val)  # map to [0,1)

    # Determine which metrics are available in ALL result dicts
    step_labels = [lbl for lbl, _ in results_list]
    dicts = [d for _, d in results_list]
    available_keys_sets = [set(d.keys()) for d in dicts]
    common_keys = set.intersection(*available_keys_sets) if available_keys_sets else set()

    metric_specs = [
        ("Lexical Drift, JSD",          "avg_lexical_drift",        colors[0], "o", "-"),
        ("Semantic Similarity, Cosine", "avg_semantic_similarity",  colors[1], "s", "--"),
        ("BERTScore F1",                "bertscore_f1",             colors[2], "^", ":"),
        ("MMD (RBF), normalized",       "mmd_rbf",                  colors[3], "D", "-."),
    ]

    # Keep only metrics present in all dicts
    metric_specs = [m for m in metric_specs if m[1] in common_keys]
    if not metric_specs:
        print("[WARN] No common metrics across all steps; nothing to plot.")
        return

    # Build series
    series = {}
    for label, key, color, marker, linestyle in metric_specs:
        vals = []
        for _, res in results_list:
            if key == "mmd_rbf":
                vals.append(_norm_mmd(res.get(key, None)))
            else:
                v = res.get(key, None)
                vals.append(float(v) if v is not None else None)

        # Replace Nones with np.nan to avoid crashes and show gaps (shouldn't happen if in common_keys)
        vals = [np.nan if v is None else v for v in vals]
        # Clamp to [0,1] for visual comparability
        vals = [min(1.0, max(0.0, v)) if np.isfinite(v) else np.nan for v in vals]
        series[label] = (vals, color, marker, linestyle)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)

    for label, (vals, color, marker, linestyle) in series.items():
        ax.plot(
            step_labels, vals,
            label=label,
            color=color,
            marker=marker,
            linewidth=lw,
            markersize=ms,
            linestyle=linestyle,
        )

    # --- Final touches ---
    ax.set_ylim(0, 1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=14 if size == "normal" else 11, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    sns.despine(ax=ax)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        print(f"Plot saved to {save_path}")
    plt.show()




def plot_drift_abrupt(results_p1, results_p2, title="Abrupt Concept Drift", save_path=None, size="normal"):
    """
    Professional plot for abrupt drift. Supports:
      - avg_lexical_drift          in [0,1] (higher = more different)
      - avg_semantic_similarity    in [0,1] (higher = more similar)
      - bertscore_f1               in [0,1] (higher = more similar)
      - mmd_rbf                    unbounded  -> normalized to [0,1] as mmd/(mmd+1)

    Missing metrics are skipped automatically.
    """

    # --- Setup for Professional Aesthetics ---
    plt.style.use('seaborn-v0_8-paper')
    colors = sns.color_palette("colorblind", 4)

    # Size presets
    if size == "small":
        figsize = (4, 3)
        font_cfg = {
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
        marker_size = 70
    else:  # normal
        figsize = (6, 5)
        font_cfg = {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
        marker_size = 120

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        **font_cfg
    })

    # --- Helper to normalize MMD and to fetch values safely ---
    def _get_metric(res, key):
        if key not in res or res[key] is None:
            return None
        val = float(res[key])
        if key == "mmd_rbf":
            # normalize to [0,1] for plotting
            val = val / (val + 1.0)
        # clamp for neatness
        return max(0.0, min(1.0, val))

    # Build the metric list dynamically based on what’s present
    candidate_metrics = [
        ("Lexical Drift, JSD",           "avg_lexical_drift"),
        ("Semantic Similarity, Cosine",  "avg_semantic_similarity"),
        ("BERTScore F1",                 "bertscore_f1"),
        ("MMD (RBF), normalized",        "mmd_rbf"),
    ]

    labels, keys = [], []
    for label, key in candidate_metrics:
        v1 = _get_metric(results_p1, key)
        v2 = _get_metric(results_p2, key)
        if v1 is not None and v2 is not None:
            labels.append(label)
            keys.append(key)

    if not keys:
        print("[WARN] No common metrics found to plot.")
        return

    x = range(len(keys))
    y1 = [_get_metric(results_p1, k) for k in keys]  # P1
    y2 = [_get_metric(results_p2, k) for k in keys]  # P2

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)

    for i, _ in enumerate(keys):
        # Arrow P1 → P2
        ax.arrow(
            i, y1[i], 0, y2[i] - y1[i],
            head_width=0.08, head_length=0.04,
            length_includes_head=True,
            color=colors[2], alpha=0.9,
            linestyle='-', linewidth=1.5,
            zorder=2
        )
        # P1 & P2 markers (professional shapes)
        ax.scatter(i, y1[i], color=colors[0], s=marker_size,
                   label="Concept 1 (P1)" if i == 0 else "", marker='s', zorder=3)
        ax.scatter(i, y2[i], color=colors[1], s=marker_size,
                   label="Concept 2 (P2)" if i == 0 else "", marker='D', zorder=3)

    # --- Final Touches ---
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0, ha="center", rotation_mode="anchor")
    ax.set_ylim(0, 1)  # shared 0–1 scale (MMD normalized)
    ax.set_ylabel("Metric Score")
    ax.set_title(title, fontsize=14 if size == "normal" else 11, fontweight='bold', pad=12)

    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    sns.despine(ax=ax)

    ax.legend(frameon=False, loc='lower right')

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {save_path}")

    plt.show()


