import re
import numpy as np
import matplotlib.pyplot as plt
from datasets import concatenate_datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import seaborn as sns
from sentence_transformers import SentenceTransformer

from Datasets import BitextTelecoDS
from DriftManager import AbruptDriftPlaceholders
from typing import List, Optional


# ----------------------------------------------------------------------
# Drift-measurement helpers
# ----------------------------------------------------------------------


def _rbf_sigma_from_ref_dists(emb_all: np.ndarray, ref_vec: np.ndarray) -> float:
    """Choose a reasonable RBF bandwidth from distances to ref."""
    d = np.linalg.norm(emb_all - ref_vec[None, :], axis=1)
    med = float(np.median(d))
    return med if med > 1e-8 else 1.0

def _rbf_k(a: np.ndarray, b: np.ndarray, sigma: float) -> float:
    return float(np.exp(-np.linalg.norm(a - b)**2 / (2.0 * sigma * sigma)))

def plot_abrupt_drift_singleline_allmetrics(
    dict_p1, dict_p2, n_samples: int = 100, smooth_w: int = 3
):
    """
    Single-line abrupt drift plot with up to four metrics:
      - Lexical Drift (JSD)
      - Semantic Drift (1 - Cosine)
      - BERTScore Drift (1 - F1)   [optional, requires bert-score]
      - MMD (RBF) normalized       [point-wise biased MMD^2 vs ref, then mmd/(1+mmd)]
    First half repeats Concept 1 text, second half repeats Concept 2 text.
    """
    keys = sorted(dict_p1.keys())
    half = n_samples // 2

    # Build texts
    ref_text = " ".join(dict_p1[k] for k in keys)
    p2_text  = " ".join(dict_p2[k] for k in keys)
    texts    = [ref_text] * half + [p2_text] * half

    # Embeddings for semantic metrics
    st = SentenceTransformer("all-MiniLM-L6-v2")
    ref_vec = st.encode([ref_text], convert_to_numpy=True)[0]
    emb_all = st.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # 1) Lexical drift per index vs ref
    lex = [lexical_jsd(t, ref_text) for t in texts]
    lex_s = rolling_mean(lex, w=smooth_w)

    # 2) Semantic drift per index vs ref (1 - cosine)
    sem = [1 - float(cosine_similarity(v[None, :], ref_vec[None, :])[0, 0]) for v in emb_all]
    sem = np.clip(sem, 0.0, 1.0).tolist()
    sem_s = rolling_mean(sem, w=smooth_w)

    # 3) BERTScore drift per index = 1 - F1 (optional)
    bert_s: Optional[List[float]] = None
    try:
        from bert_score import score as bertscore_score
        refs  = [ref_text] * n_samples
        P, R, F1 = bertscore_score(
            cands=texts, refs=refs,
            model_type="microsoft/deberta-base-mnli",
            lang="en", verbose=False, rescale_with_baseline=True
        )
        bert = (1.0 - F1.numpy()).tolist()
        # clamp to [0,1] for safety
        bert = [max(0.0, min(1.0, x)) for x in bert]
        bert_s = rolling_mean(bert, w=smooth_w)
    except Exception as e:
        print(f"[INFO] BERTScore unavailable, skipping: {e}")

    # 4) Point-wise biased MMD^2 vs ref, normalized to [0,1]
    sigma = _rbf_sigma_from_ref_dists(emb_all, ref_vec)
    mmd_norm = []
    for v in emb_all:
        k_xx = 1.0
        k_yy = 1.0
        k_xy = _rbf_k(v, ref_vec, sigma)
        mmd2 = (k_xx + k_yy - 2.0 * k_xy)  # biased MMD^2 for m=n=1
        mmd_norm.append(mmd2 / (mmd2 + 1.0))
    mmd_s = rolling_mean(mmd_norm, w=smooth_w)

    # --- Plot all available lines
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(lex_s, label="Lexical Drift (JSD)")
    ax.plot(sem_s, label="Semantic Drift (1 − Cosine)")
    if bert_s is not None:
        ax.plot(bert_s, label="BERTScore Drift (1 − F1)")
    ax.plot(mmd_s, label="MMD (RBF) normalized")

    ax.axvline(half, color="k", linestyle="--", linewidth=1.2, label="Drift point")
    ax.set_xlim(0, n_samples)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Drift Score")
    ax.set_title("Abrupt Drift — Single-Line View with Four Metrics", fontweight="bold")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def jsd(p, q, eps=1e-12):
    """Jensen–Shannon distance between two probability vectors."""
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def kl(a, b):
        a, b = np.clip(a, eps, 1), np.clip(b, eps, 1)
        return np.sum(a * np.log(a / b))

    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def lexical_jsd(a: str, b: str) -> float:
    """Lexical drift via JSD on bag-of-words, robust to empty tokens."""
    a, b = a.strip(), b.strip()
    if not a or not b:
        return 0.0

    vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    try:
        X = vect.fit_transform([a, b]).toarray().astype(float)
    except ValueError:
        # Handles empty vocab after tokenization
        return 0.0

    if X.sum() == 0:
        return 0.0

    return jsd(X[0], X[1])


def rolling_mean(x, w=21):
    """Simple rolling mean for smooth line plots."""
    if w <= 1:
        return np.asarray(x, float)
    k = np.ones(w) / w
    return np.convolve(np.asarray(x, float), k, mode="same")


# ----------------------------------------------------------------------
# Pairwise placeholder drift (metric view)
# ----------------------------------------------------------------------

def plot_abrupt_drift_placeholders(dict_p1, dict_p2):
    """
    Compute lexical & semantic drift between corresponding placeholders
    in Concept 1 (P1) and Concept 2 (P2), and visualize the jump.
    """
    keys = sorted(set(dict_p1.keys()) & set(dict_p2.keys()))
    vals_p1 = [dict_p1[k] for k in keys]
    vals_p2 = [dict_p2[k] for k in keys]

    # --- Lexical drift per pair
    lex_drift = [lexical_jsd(v1, v2) for v1, v2 in zip(vals_p1, vals_p2)]

    # --- Semantic drift per pair
    st = SentenceTransformer("all-MiniLM-L6-v2")
    emb_p1 = st.encode(vals_p1, convert_to_numpy=True, show_progress_bar=False)
    emb_p2 = st.encode(vals_p2, convert_to_numpy=True, show_progress_bar=False)
    sem_drift = [1 - float(cosine_similarity(e1[None, :], e2[None, :])[0, 0])
                 for e1, e2 in zip(emb_p1, emb_p2)]

    # --- Smooth lines for readability
    lex_drift_s = rolling_mean(lex_drift, w=3)
    sem_drift_s = rolling_mean(sem_drift, w=3)

    # --- Plot metric drift lines
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lex_drift_s, label="Lexical Drift (JSD)", color="tab:blue")
    ax.plot(sem_drift_s, label="Semantic Drift (1 − Cosine)", color="tab:orange")
    ax.set_xlim(0, len(keys) - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Placeholder Index")
    ax.set_ylabel("Drift Score")
    ax.set_title("Abrupt Drift across Placeholder Pairs (Concept 1 → Concept 2)",
                 fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Optional: return for further use
    return dict(keys=keys, lex_drift=lex_drift, sem_drift=sem_drift,
                emb_p1=emb_p1, emb_p2=emb_p2)


# ----------------------------------------------------------------------
# Placeholder-pair drift in semantic space (vector view)
# ----------------------------------------------------------------------

def plot_placeholder_pair_pca(emb_p1, emb_p2, keys):
    """
    Visualize placeholder drift as vector arrows in 2-D PCA space.
    Each arrow shows how one placeholder moves from Concept 1 to Concept 2.
    """
    pca = PCA(n_components=2)
    combined = np.vstack([emb_p1, emb_p2])
    proj = pca.fit_transform(combined)
    p1_proj, p2_proj = proj[:len(emb_p1)], proj[len(emb_p1):]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(p1_proj[:, 0], p1_proj[:, 1], color="tab:blue",
               label="Concept 1 (P1)", s=60, marker="o", alpha=0.8)
    ax.scatter(p2_proj[:, 0], p2_proj[:, 1], color="tab:orange",
               label="Concept 2 (P2)", s=60, marker="^", alpha=0.8)

    # Draw drift arrows
    for i in range(len(keys)):
        ax.arrow(p1_proj[i, 0], p1_proj[i, 1],
                 p2_proj[i, 0] - p1_proj[i, 0],
                 p2_proj[i, 1] - p1_proj[i, 1],
                 color="gray", alpha=0.5, head_width=0.03,
                 length_includes_head=True)

    ax.set_title("Placeholder-level Abrupt Concept Drift (PCA Projection)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Single-line abrupt drift view
# ----------------------------------------------------------------------

def plot_abrupt_drift_singleline(dict_p1, dict_p2, n_samples=100):
    """
    Show abrupt drift as a single continuous line:
    first half Concept 1 (identical, low drift), second half Concept 2 (changed).
    """
    keys = sorted(dict_p1.keys())
    half = n_samples // 2
    st = SentenceTransformer("all-MiniLM-L6-v2")

    # reference = Concept 1 concatenation
    ref_text = " ".join(dict_p1[k] for k in keys)
    ref_vec = st.encode([ref_text], convert_to_numpy=True)[0]

    # half1: concept 1 copies (no drift)
    texts_p1 = [ref_text] * half
    # half2: concept 2 copies (drifted)
    text_p2 = " ".join(dict_p2[k] for k in keys)
    texts_p2 = [text_p2] * half

    # combine
    all_texts = texts_p1 + texts_p2
    emb = st.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)

    # compute drift = 1 - cosine similarity to ref
    drift = [1 - float(cosine_similarity(v[None, :], ref_vec[None, :])[0, 0]) for v in emb]
    drift_s = rolling_mean(drift, w=3)

    # plot
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(drift_s, color="tab:orange", label="Semantic Drift (1 − Cosine)")
    ax.axvline(half, color="k", linestyle="--", linewidth=1.2, label="Drift point")
    ax.set_xlim(0, n_samples)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Drift Score")
    ax.set_title("Abrupt Drift (Single-Line Representation)", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore_score
import torch

# --- helpers ---
def jsd(p, q, eps=1e-12):
    """Jensen–Shannon distance between two probability vectors."""
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    def kl(a, b):
        a, b = np.clip(a, eps, 1), np.clip(b, eps, 1)
        return np.sum(a * np.log(a / b))
    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))

def lexical_jsd(a, b):
    """Lexical drift (bag-of-words JSD)."""
    vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vect.fit_transform([a, b]).toarray().astype(float)
    return jsd(X[0], X[1])

def mmd_rbf(X, Y, sigma=None):
    """Unbiased MMD with RBF kernel."""
    if sigma is None:
        sigma = np.median(np.linalg.norm(X[:, None] - Y[None, :], axis=2))
    XX = np.exp(-np.square(np.linalg.norm(X[:, None] - X[None, :], axis=2)) / (2 * sigma**2))
    YY = np.exp(-np.square(np.linalg.norm(Y[:, None] - Y[None, :], axis=2)) / (2 * sigma**2))
    XY = np.exp(-np.square(np.linalg.norm(X[:, None] - Y[None, :], axis=2)) / (2 * sigma**2))
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

def rolling_mean(x, w=7):
    if w <= 1: return np.asarray(x, float)
    k = np.ones(w) / w
    return np.convolve(np.asarray(x, float), k, mode="same")

# --- main ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore_score
import torch

# ---------- Helper Functions ----------
def jsd(p, q, eps=1e-12):
    """Jensen–Shannon distance between two probability vectors."""
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    def kl(a, b):
        a, b = np.clip(a, eps, 1), np.clip(b, eps, 1)
        return np.sum(a * np.log(a / b))
    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))

def lexical_jsd(a, b):
    """Lexical drift (bag-of-words JSD)."""
    vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vect.fit_transform([a, b]).toarray().astype(float)
    return jsd(X[0], X[1])

def mmd_rbf(X, Y, sigma=None):
    """Unbiased MMD with RBF kernel."""
    if sigma is None:
        sigma = np.median(np.linalg.norm(X[:, None] - Y[None, :], axis=2))
    XX = np.exp(-np.square(np.linalg.norm(X[:, None] - X[None, :], axis=2)) / (2 * sigma**2))
    YY = np.exp(-np.square(np.linalg.norm(Y[:, None] - Y[None, :], axis=2)) / (2 * sigma**2))
    XY = np.exp(-np.square(np.linalg.norm(X[:, None] - Y[None, :], axis=2)) / (2 * sigma**2))
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

def rolling_mean(x, w=7):
    """Simple rolling mean smoothing."""
    if w <= 1:
        return np.asarray(x, float)
    k = np.ones(w) / w
    return np.convolve(np.asarray(x, float), k, mode="same")

# ---------- Main Function ----------
def plot_abrupt_drift_four_metrics(test_data_p1p2, text_field="response", n_samples=None):
    """
    Plots a single continuous line per metric showing the abrupt shift
    between Concept 1 (p1) and Concept 2 (p2).
    Includes Lexical Drift, Semantic Drift, BERTScore Drift, and MMD Drift.
    """
    if n_samples is not None:
        test_data_p1p2 = test_data_p1p2.select(range(min(n_samples, len(test_data_p1p2))))

    n = len(test_data_p1p2)
    half = n // 2
    print(f"Total samples: {n} (Concept1: {half}, Concept2: {n-half})")

    texts = test_data_p1p2[text_field]
    texts_p1 = texts[:half]
    texts_p2 = texts[half:]

    st = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode both halves
    emb_p1 = st.encode(texts_p1, convert_to_numpy=True, show_progress_bar=False)
    emb_p2 = st.encode(texts_p2, convert_to_numpy=True, show_progress_bar=False)
    mean_p1 = np.mean(emb_p1, axis=0)
    mean_p2 = np.mean(emb_p2, axis=0)

    # --- 1. Lexical Drift (JSD between matching indices) ---
    lex_p1 = [lexical_jsd(a, b) for a, b in zip(texts_p1, texts_p1)]
    lex_p2 = [lexical_jsd(a, b) for a, b in zip(texts_p2, texts_p2)]
    drift_lex = rolling_mean(lex_p1 + lex_p2, w=5)

    # --- 2. Semantic Drift (1 - cosine to own concept mean) ---
    drift_sem = [1 - float(cosine_similarity(v[None, :], mean_p1[None, :])[0, 0]) for v in emb_p1] + \
                [1 - float(cosine_similarity(v[None, :], mean_p2[None, :])[0, 0]) for v in emb_p2]
    drift_sem = rolling_mean(drift_sem, w=5)

    # --- 3. BERTScore Drift (1 - F1) ---
    try:
        P, R, F1_p1 = bertscore_score(
            cands=texts_p1, refs=texts_p1,
            model_type="microsoft/deberta-base-mnli",
            lang="en",                      # ✅ required fix
            verbose=False, rescale_with_baseline=True
        )
        P, R, F1_p2 = bertscore_score(
            cands=texts_p2, refs=texts_p2,
            model_type="microsoft/deberta-base-mnli",
            lang="en",                      # ✅ required fix
            verbose=False, rescale_with_baseline=True
        )
        drift_bert = rolling_mean(list(1 - F1_p1.numpy()) + list(1 - F1_p2.numpy()), w=5)
    except Exception as e:
        print(f"[WARN] BERTScore computation failed: {e}")
        drift_bert = [0.0] * n

    # --- 4. Distributional Drift (MMD normalized) ---
    mmd_val = mmd_rbf(emb_p1, emb_p2)
    mmd_norm = mmd_val / (mmd_val + 1.0)
    drift_mmd = [0.0] * half + [mmd_norm] * half

    # --- Normalize each metric to [0, 1] for fair comparison ---
    def normalize(x):
        x = np.array(x)
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

    drift_lex = normalize(drift_lex)
    drift_sem = normalize(drift_sem)
    drift_bert = normalize(drift_bert)
    drift_mmd = normalize(drift_mmd)

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(drift_lex, label="Lexical Drift (JSD)", color="tab:blue", linewidth=1.6)
    ax.plot(drift_sem, label="Semantic Drift (1 − Cosine)", color="tab:orange", linewidth=1.6)
    ax.plot(drift_bert, label="BERTScore Drift (1 − F1)", color="tab:green", linewidth=1.6)
    ax.plot(drift_mmd, label="Distributional Drift (MMD)", color="tab:red", linewidth=1.6)

    ax.axvline(half, color="k", linestyle="--", linewidth=1.2, label="Drift Point (p1 → p2)")
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Normalized Drift Score")
    ax.set_title("Abrupt Concept Drift — Four Metrics", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    print("✅ Plot complete. Abrupt drift (four metrics) visualized successfully.")




# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main1():
    print("Generating single-line abrupt drift with all metrics...")
    plot_abrupt_drift_singleline_allmetrics(
        AbruptDriftPlaceholders.PLACEHOLDER_VALUES_P1,
        AbruptDriftPlaceholders.PLACEHOLDER_VALUES_P2,
        n_samples=100,  # or 15600 in your full run
        smooth_w=3
    )


def main2():


    # 4. Load the Dataset - Train, Test, Validation
    train_ratio = 0.6
    test_ratio = 0.3
    val_ratio = 0.1
    dataset = BitextTelecoDS.get_bitext_telecom_dataset_splits(train_ratio=train_ratio, test_ratio=test_ratio,
                                                               val_ratio=val_ratio)
    train_data = dataset["train"]
    test_data = dataset["test"]
    val_data = dataset["validation"]
    combined_data = dataset["combined"]

    # #todo: remove this on production
    train_data = train_data.select(range(200))
    test_data = test_data.select(range(200))
    val_data = val_data.select(range(200))
    combined_data = concatenate_datasets([train_data, test_data])

    train_data_p1 = BitextTelecoDS.preprocess_bitext_ds(train_data, "p1", "abrupt")
    val_data_p1 = BitextTelecoDS.preprocess_bitext_ds(val_data, "p1", "abrupt")
    combined_data_p1 = BitextTelecoDS.preprocess_bitext_ds(combined_data, "p1", "abrupt")
    combined_data_p2 = BitextTelecoDS.preprocess_bitext_ds(combined_data, "p2", "abrupt")

    # Split test into two halves
    n_test = len(test_data)
    half = n_test // 2

    test_data_p1 = BitextTelecoDS.preprocess_bitext_ds(test_data.select(range(half)), placeholder="p1",
                                                       drift_type="abrupt")
    test_data_p2 = BitextTelecoDS.preprocess_bitext_ds(test_data.select(range(half, n_test)), placeholder="p2",
                                                       drift_type="abrupt")
    # Merge them back
    test_data_p1p2 = concatenate_datasets([test_data_p1, test_data_p2])  # preserve their sequence.
    test_data_p1p2 = concatenate_datasets([test_data_p1, test_data_p2])
    print("✅ Test dataset with abrupt drift prepared successfully.")

    plot_abrupt_drift_four_metrics(
        test_data_p1p2,
        text_field="response",
        n_samples=200  # optional for faster test
    )


if __name__ == "__main__":
    # main1()
    main2()
