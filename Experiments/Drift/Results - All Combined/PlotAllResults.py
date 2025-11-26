import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# Build dataframe
# -------------------------
data = {
    "drift": ["Abrupt"] * 3 + ["Incremental"] * 3 + ["Gradual"] * 3,
    "Strategy": [
        "Greedy", "Temp Scaled", "ODD",
        "Greedy", "Temp Scaled", "ODD",
        "Greedy", "Temp Scaled", "ODD",
    ],
    "Exact Match": [0, 0, 0.096, 0, 0, 0.052, 0, 0, 0.037],
    "Edit Distance": [0.759, 0.747, 0.796, 0.740, 0.725, 0.803, 0.702, 0.687, 0.790],
    "BLEU": [0.652, 0.631, 0.679, 0.600, 0.574, 0.682, 0.555, 0.529, 0.663],
    "ROUGE-L": [0.803, 0.790, 0.828, 0.790, 0.775, 0.838, 0.760, 0.745, 0.825],
    "Cosine Similarity": [0.865, 0.863, 0.968, 0.865, 0.862, 0.976, 0.855, 0.853, 0.971],
    "ChrF": [80.374, 79.199, 83.147, 78.304, 76.886, 83.826, 74.031, 72.710, 82.824],
    "BERTScore": [0.911, 0.907, 0.928, 0.908, 0.902, 0.935, 0.890, 0.885, 0.928],
}

df = pd.DataFrame(data)

# Normalize ChrF
df["ChrF"] = df["ChrF"] / 100.0

# Melt
plot_df = df.melt(
    id_vars=["Strategy", "drift"],
    var_name="metric",
    value_name="score"
)

# -------------------------
# Plot: One row of blocks (7 metrics), each showing the 3 drifts
# -------------------------

metrics = ["Exact Match", "Edit Distance", "BLEU", "ROUGE-L",
           "Cosine Similarity", "ChrF", "BERTScore"]

fig, axes = plt.subplots(
    nrows=1, ncols=len(metrics),
    figsize=(22, 5),
    sharey=True
)

palette = sns.color_palette("Set2", 3)

for i, metric in enumerate(metrics):
    ax = axes[i]

    sub = plot_df[plot_df["metric"] == metric]

    sns.barplot(
        data=sub,
        x="drift",
        y="score",
        hue="Strategy",
        ax=ax,
        palette=palette,
        width=0.55
    )

    ax.set_title(metric, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Mean Score" if i == 0 else "")
    ax.tick_params(axis="x", rotation=0, labelsize=9)

    if i == 0:
        ax.legend(title="Strategy", fontsize=7)
    else:
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig("nlg_metrics_across_drifts.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: nlg_metrics_across_drifts.png")
