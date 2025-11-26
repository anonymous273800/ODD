import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from matplotlib.lines import Line2D


# ---------- helpers ----------
def jsd(p, q, eps=1e-12):
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def kl(a, b):
        a, b = np.clip(a, eps, 1), np.clip(b, eps, 1)
        return np.sum(a * np.log(a / b))

    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def lexical_jsd_safe(a: str, b: str) -> float:
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return 0.0

    vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    try:
        X = vect.fit_transform([a, b]).toarray().astype(float)
    except ValueError:
        return 0.0

    if X.sum() == 0:
        return 0.0

    return jsd(X[0], X[1])


def rbf_sigma_from_set(Y: np.ndarray) -> float:
    D = np.linalg.norm(Y[:, None] - Y[None, :], axis=2)
    med = float(np.median(D[D > 0])) if np.any(D > 0) else 1.0
    return med if med > 1e-8 else 1.0


def mean_rbf_sim_to_set(x: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    d2 = np.sum((Y - x[None, :]) ** 2, axis=1)
    return float(np.mean(np.exp(-d2 / (2.0 * sigma * sigma))))


def minmax_norm(a, b):
    """Normalize two arrays together to [0,1]."""
    z = np.concatenate([a, b])
    lo, hi = float(np.min(z)), float(np.max(z))
    rng = hi - lo if hi > lo else 1.0
    return (np.asarray(a) - lo) / rng, (np.asarray(b) - lo) / rng


# ---------- main plotting ----------
def plot_placeholder_drift_lines_anchored(dict_p1, dict_p2):
    keys = sorted(set(dict_p1.keys()) & set(dict_p2.keys()))
    vals_p1 = [dict_p1[k] for k in keys]
    vals_p2 = [dict_p2[k] for k in keys]
    n = len(keys)
    x = np.arange(n)

    # --- build anchors from Concept 1 ---
    ref_text_p1 = " ".join(vals_p1)
    st = SentenceTransformer("all-MiniLM-L6-v2")

    emb_p1 = st.encode(vals_p1, convert_to_numpy=True, show_progress_bar=False)
    emb_p2 = st.encode(vals_p2, convert_to_numpy=True, show_progress_bar=False)
    mean_p1 = np.mean(emb_p1, axis=0)

    # ---- metrics ----
    lex_p1 = [lexical_jsd_safe(v, ref_text_p1) for v in vals_p1]
    lex_p2 = [lexical_jsd_safe(v, ref_text_p1) for v in vals_p2]
    lex_p1_n, lex_p2_n = minmax_norm(lex_p1, lex_p2)

    sem_p1 = [1 - float(cosine_similarity(v[None, :], mean_p1[None, :])[0, 0]) for v in emb_p1]
    sem_p2 = [1 - float(cosine_similarity(v[None, :], mean_p1[None, :])[0, 0]) for v in emb_p2]
    sem_p1_n, sem_p2_n = minmax_norm(sem_p1, sem_p2)

    # ---- BERTScore ----
    try:
        from bert_score import score as bertscore_score

        refs = [ref_text_p1] * n
        _, _, F1_p1 = bertscore_score(
            cands=vals_p1, refs=refs,
            model_type="microsoft/deberta-base-mnli",
            lang="en", rescale_with_baseline=True, verbose=False
        )
        _, _, F1_p2 = bertscore_score(
            cands=vals_p2, refs=refs,
            model_type="microsoft/deberta-base-mnli",
            lang="en", rescale_with_baseline=True, verbose=False
        )

        bert_p1 = (1.0 - F1_p1.numpy()).tolist()
        bert_p2 = (1.0 - F1_p2.numpy()).tolist()

    except Exception as e:
        print(f"[WARN] BERTScore unavailable, fallback to zeros: {e}")
        bert_p1 = [0.0] * n
        bert_p2 = [0.0] * n

    bert_p1_n, bert_p2_n = minmax_norm(bert_p1, bert_p2)

    sigma = rbf_sigma_from_set(emb_p1)
    mmdlike_p1 = [1 - mean_rbf_sim_to_set(v, emb_p1, sigma) for v in emb_p1]
    mmdlike_p2 = [1 - mean_rbf_sim_to_set(v, emb_p1, sigma) for v in emb_p2]
    mmd_p1_n, mmd_p2_n = minmax_norm(mmdlike_p1, mmdlike_p2)

    panels = [
        ("Lexical Similarity (JSD)", lex_p1_n, lex_p2_n),
        ("Semantic Similarity (1 − Cosine)", sem_p1_n, sem_p2_n),
        ("BERTScore (1 − F1)", bert_p1_n, bert_p2_n),
        ("Kernel Similarity (RBF mean sim)", mmd_p1_n, mmd_p2_n),
    ]

    # ---------- plotting ----------
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), facecolor="white")
    axs = axs.ravel()

    for i, (title, y_blue, y_green) in enumerate(panels):
        ax = axs[i]

        ax.set_facecolor("#f9f9f9")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.plot(x, y_blue, color="tab:blue", marker="o", linewidth=1.2,
                markersize=7, markeredgecolor="white", zorder=3)

        ax.plot(x, y_green, color="tab:green", marker="o", linewidth=1.2,
                markersize=7, markeredgecolor="white", zorder=3)

        for xi, b, g in zip(x, y_blue, y_green):
            ax.plot([xi, xi], [b, g], color="lightgray", alpha=0.35, linewidth=0.6)

        ax.grid(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))
        ax.yaxis.set_major_locator(plt.MaxNLocator(8))
        ax.set_aspect('auto')

        ax.set_xlabel(title, fontsize=14, fontweight="bold", labelpad=8)
        ax.set_xticklabels([])

    # Legend
    proxy_handles = [
        Line2D([0], [0], color="tab:blue", lw=3.5, marker="o",
               markersize=10, markeredgecolor="white"),
        Line2D([0], [0], color="tab:green", lw=3.5, marker="o",
               markersize=10, markeredgecolor="white"),
    ]
    proxy_labels = ["Concept 1 (C1)", "Concept 2 (C2)"]

    leg = fig.legend(
        proxy_handles,
        proxy_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=14,
        ncol=2,
    )

    for text in leg.get_texts():
        text.set_fontweight("bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ---------- main ----------
def main():
    from DriftManager import AbruptDriftPlaceholders
    dict_p1 = AbruptDriftPlaceholders.PLACEHOLDER_VALUES_P1
    dict_p2 = AbruptDriftPlaceholders.PLACEHOLDER_VALUES_P2
    plot_placeholder_drift_lines_anchored(dict_p1, dict_p2)


if __name__ == "__main__":
    main()
