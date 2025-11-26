import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


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


def minmax_norm_multi(arrs):
    """Normalize a list of arrays together to [0,1]."""
    z = np.concatenate(arrs)
    lo, hi = float(np.min(z)), float(np.max(z))
    rng = hi - lo if hi > lo else 1.0
    return [(np.asarray(a) - lo) / rng for a in arrs]


# ---------- main plotting ----------
def plot_placeholder_drift_lines_anchored(*dicts):
    n_concepts = len(dicts)
    assert n_concepts >= 2, "At least two concept dicts are required."

    # intersection of all placeholder keys
    common_keys = sorted(set.intersection(*(set(d.keys()) for d in dicts)))
    vals_all = [[d[k] for k in common_keys] for d in dicts]
    n = len(common_keys)
    x = np.arange(n)

    # --- reference (first concept) ---
    ref_text = " ".join(vals_all[0])
    st = SentenceTransformer("all-MiniLM-L6-v2")
    emb_all = [st.encode(vals, convert_to_numpy=True, show_progress_bar=False) for vals in vals_all]
    mean_ref = np.mean(emb_all[0], axis=0)

    # ---- metrics ----
    # Lexical JSD
    lex_all = [[lexical_jsd_safe(v, ref_text) for v in vals] for vals in vals_all]
    lex_norm = minmax_norm_multi(lex_all)

    # Semantic (1 - cosine)
    sem_all = [[1 - float(cosine_similarity(v[None, :], mean_ref[None, :])[0, 0])
                for v in emb] for emb in emb_all]
    sem_norm = minmax_norm_multi(sem_all)

    # BERTScore (1 - F1)
    try:
        from bert_score import score as bertscore_score
        refs = [ref_text] * n
        bert_all = []
        for vals in vals_all:
            _, _, F1 = bertscore_score(
                cands=vals, refs=refs,
                model_type="microsoft/deberta-base-mnli",
                lang="en", rescale_with_baseline=True, verbose=False
            )
            bert_all.append((1.0 - F1.numpy()).tolist())
        bert_norm = minmax_norm_multi(bert_all)
    except Exception as e:
        print(f"[WARN] BERTScore unavailable, fallback to zeros: {e}")
        bert_norm = [[0.0] * n for _ in range(n_concepts)]

    # Kernel similarity (MMD-like)
    sigma = rbf_sigma_from_set(emb_all[0])
    mmd_all = [[1 - mean_rbf_sim_to_set(v, emb_all[0], sigma) for v in emb]
               for emb in emb_all]
    mmd_norm = minmax_norm_multi(mmd_all)

    panels = [
        ("Lexical Similarity (JSD)", lex_norm),
        ("Semantic Similarity (1 − Cosine)", sem_norm),
        ("BERTScore (1 − F1)", bert_norm),
        ("Kernel Similarity (RBF mean sim)", mmd_norm),
    ]

    colors = plt.cm.tab10.colors[:n_concepts]
    labels = [f"Concept {i+1} (C{i+1})" for i in range(n_concepts)]

    # ---------- plotting ----------
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), facecolor="white")
    axs = axs.ravel()

    for i, (title, metrics) in enumerate(panels):
        ax = axs[i]
        ax.set_facecolor("#f9f9f9")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for c, y, label in zip(colors, metrics, labels):
            ax.plot(x, y, color=c, marker="o", linewidth=1.2,
                    markersize=5, markeredgecolor="white", zorder=3, label=label)

        ax.grid(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))
        ax.yaxis.set_major_locator(plt.MaxNLocator(8))
        ax.set_xlabel(title, fontsize=14,fontweight="bold", labelpad=8)
        ax.set_xticklabels([])

    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=n_concepts,
    #            frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.01))

    from matplotlib.lines import Line2D

    # --------- PROXY HANDLES FOR BOLD LEGEND LINES ----------
    proxy_handles = [
        Line2D(
            [0], [0],
            color=colors[i],
            lw=3.5,  # bold line
            marker="o",
            markersize=10,  # bigger markers
            markeredgecolor="white"
        )
        for i in range(n_concepts)
    ]

    leg = fig.legend(
        proxy_handles,
        labels,
        loc="upper center",
        ncol=n_concepts,
        frameon=False,
        fontsize=12,  # larger legend font
        bbox_to_anchor=(0.5, 1.01)
    )

    # Bold legend text
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ---------- main ----------
def main():
    from DriftManager import GradualDriftPlaceholders
    dict_p1 = GradualDriftPlaceholders.PLACEHOLDER_VALUES_P1
    dict_p2 = GradualDriftPlaceholders.PLACEHOLDER_VALUES_P2
    dict_p3 = GradualDriftPlaceholders.PLACEHOLDER_VALUES_P3
    dict_p4 = GradualDriftPlaceholders.PLACEHOLDER_VALUES_P4
    dict_p5 = GradualDriftPlaceholders.PLACEHOLDER_VALUES_P5
    dict_p6 = GradualDriftPlaceholders.PLACEHOLDER_VALUES_P6
    plot_placeholder_drift_lines_anchored(dict_p1, dict_p2, dict_p3, dict_p4, dict_p5, dict_p6)


if __name__ == "__main__":
    main()
