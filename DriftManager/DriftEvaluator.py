import numpy as np
from typing import Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from DriftManager import (
    AbruptDriftPlaceholders,
    IncrementalDriftPlaceholders,
    GradualDriftPlaceholders,
    DriftPlotter,
)

# -------------------------------------------------
# Basic utilities
# -------------------------------------------------

def jaccard_distance(s1: str, s2: str) -> float:
    """Token-level Jaccard distance (1 - Jaccard similarity) on whitespace tokens."""
    set1, set2 = set(s1.lower().split()), set(s2.lower().split())
    if not set1 and not set2:
        return 0.0
    return 1.0 - len(set1 & set2) / len(set1 | set2)


def _pairwise_jsd_or_jaccard(t1: str, t2: str) -> float:
    """
    Adaptive lexical distance:
      - Jaccard for short strings (fewer than 5 tokens in either text)
      - JSD on bag-of-words distributions for longer strings
    """
    if len(t1.split()) < 5 or len(t2.split()) < 5:
        return jaccard_distance(t1, t2)
    vec = CountVectorizer()
    X1 = vec.fit_transform([t1])
    X2 = vec.transform([t2])
    p = np.asarray(X1.sum(axis=0)).ravel()
    q = np.asarray(X2.sum(axis=0)).ravel()
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q))


# -------------------------------------------------
# MMD (RBF kernel) helpers
# -------------------------------------------------

def _median_heuristic_sigma(X: np.ndarray) -> float:
    """
    Median heuristic for RBF bandwidth using pairwise squared distances.
    Falls back to 1.0 if degenerate.
    """
    diffs = X[:, None, :] - X[None, :, :]
    D2 = np.sum(diffs * diffs, axis=-1)
    D2_nonzero = D2[D2 > 0]
    med = np.median(D2_nonzero) if D2_nonzero.size > 0 else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return float(np.sqrt(med / 2.0))


def _rbf_kernel(d2: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-d2 / (2.0 * sigma * sigma))


def _mmd_rbf_unbiased(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    """
    Unbiased MMD^2 with RBF kernel between sets X and Y, then returns sqrt(MMD^2) as a distance.
    X: (n, d), Y: (m, d)
    """
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return 0.0  # not enough samples for unbiased estimate

    if sigma is None:
        sigma = _median_heuristic_sigma(np.vstack([X, Y]))

    # Pairwise squared distances
    XX = X[:, None, :] - X[None, :, :]
    YY = Y[:, None, :] - Y[None, :, :]
    XY = X[:, None, :] - Y[None, :, :]
    Dxx = np.sum(XX * XX, axis=-1)
    Dyy = np.sum(YY * YY, axis=-1)
    Dxy = np.sum(XY * XY, axis=-1)

    Kxx = _rbf_kernel(Dxx, sigma)
    Kyy = _rbf_kernel(Dyy, sigma)
    Kxy = _rbf_kernel(Dxy, sigma)

    # Unbiased: exclude diagonal terms
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = (Kxx.sum() / (n * (n - 1) + 1e-12)
            + Kyy.sum() / (m * (m - 1) + 1e-12)
            - 2.0 * Kxy.mean())

    mmd2 = max(0.0, float(mmd2))  # numerical safety
    return float(np.sqrt(mmd2))


# -------------------------------------------------
# Main metric function (now with optional BERTScore & MMD)
# -------------------------------------------------

def measure_placeholders_drift(
    dict_p1,
    dict_p2,
    embedding_model: str = "all-MiniLM-L6-v2",
    precision: int = 4,
    use_bert_score: bool = True,
    bertscore_model: str = "microsoft/deberta-base-mnli",  # solid, lighter than xlarge
    compute_mmd: bool = True,
):
    """
    Measure across all matching placeholder keys:

      * avg_lexical_drift       : Adaptive lexical distance (Jaccard for short, JSD for long), higher = more different.
      * avg_semantic_similarity : Cosine similarity on sentence embeddings, higher = more similar (1 = identical).

    Optional:
      * bertscore_f1 : Average BERTScore F1 (higher = more similar). Requires: pip install bert-score
      * mmd_rbf      : Set-level distributional drift between the two embedding sets (higher = more different).

    Returns a dict with rounded metrics (to `precision` decimals).
    """
    # Align keys for stable order
    common_keys = sorted(set(dict_p1.keys()) & set(dict_p2.keys()))
    if not common_keys:
        results = {
            "avg_lexical_drift": 0.0,
            "avg_semantic_similarity": 1.0,
        }
        if use_bert_score:
            results["bertscore_f1"] = 1.0
        if compute_mmd:
            results["mmd_rbf"] = 0.0
        return {k: round(v, precision) for k, v in results.items()}

    texts1 = [dict_p1[k] for k in common_keys]
    texts2 = [dict_p2[k] for k in common_keys]

    # --- Lexical (pairwise, then average)
    lexical = [_pairwise_jsd_or_jaccard(t1, t2) for t1, t2 in zip(texts1, texts2)]
    avg_lexical = float(np.mean(lexical)) if lexical else 0.0

    # --- Embeddings (batched)
    st_model = SentenceTransformer(embedding_model)
    emb1 = st_model.encode(texts1, convert_to_numpy=True, show_progress_bar=False)
    emb2 = st_model.encode(texts2, convert_to_numpy=True, show_progress_bar=False)

    # Pairwise cosine similarity (aligned keys); clamp to [0,1]
    sims = []
    for v1, v2 in zip(emb1, emb2):
        sim = float(cosine_similarity(v1[None, :], v2[None, :])[0][0])
        sim = max(0.0, min(1.0, sim))
        sims.append(sim)
    avg_sim = float(np.mean(sims)) if sims else 1.0

    # --- Optional: BERTScore-F1
    bert_f1 = None
    if use_bert_score:
        try:
            from bert_score import score as bertscore_score
            P, R, F1 = bertscore_score(
                cands=texts2, refs=texts1,
                model_type=bertscore_model, lang="en",
                verbose=False, rescale_with_baseline=True
            )
            bert_f1 = float(F1.mean().item())
        except Exception as e:
            # Keep going even if BERTScore isn't installed or errors out
            bert_f1 = None
            print(f"[WARN] BERTScore not computed: {e}")

    # --- Optional: MMD on embeddings (set-level)
    mmd_val = None
    if compute_mmd:
        mmd_val = _mmd_rbf_unbiased(emb1, emb2, sigma=None)

    # --- Collect & round
    results = {
        "avg_lexical_drift": avg_lexical,
        "avg_semantic_similarity": avg_sim,
    }
    if use_bert_score and bert_f1 is not None:
        results["bertscore_f1"] = bert_f1
    if compute_mmd and mmd_val is not None:
        results["mmd_rbf"] = mmd_val

    results = {k: round(v, precision) for k, v in results.items()}
    return results


# -------------------------------------------------
# Example usage (your original main, unchanged API)
# -------------------------------------------------

if __name__ == "__main__":
    print("Evaluate Drift for Abrupt Drift Placeholders")

    # P1 baseline (self): lexical ~0.0, semantic similarity ~1.0, mmd ~0.0
    results_p1 = measure_placeholders_drift(
        AbruptDriftPlaceholders.PLACEHOLDER_VALUES_P1,
        AbruptDriftPlaceholders.PLACEHOLDER_VALUES_P1
    )

    # P1 → P2 drift
    results_p2 = measure_placeholders_drift(
        AbruptDriftPlaceholders.PLACEHOLDER_VALUES_P1,
        AbruptDriftPlaceholders.PLACEHOLDER_VALUES_P2
    )

    print("Abrupt Drift - P1 baseline:", results_p1)
    print("Abrupt Drift - P1 → P2:", results_p2)

    DriftPlotter.plot_drift_abrupt(results_p1, results_p2)
    print("--- --- --- ---")
    ######################################################################################

    # # Uncomment to run incremental drift block with the same metrics:
    # print("Evaluate Drift for Incremental Drift Placeholders")
    # placeholders = [
    #     IncrementalDriftPlaceholders.PLACEHOLDER_VALUES_P1,
    #     IncrementalDriftPlaceholders.PLACEHOLDER_VALUES_P2,
    #     IncrementalDriftPlaceholders.PLACEHOLDER_VALUES_P3,
    #     IncrementalDriftPlaceholders.PLACEHOLDER_VALUES_P4,
    #     IncrementalDriftPlaceholders.PLACEHOLDER_VALUES_P5,
    # ]
    #
    # results_list = []
    #
    # # # Total P1→P1
    # res_base = measure_placeholders_drift(
    #     placeholders[0],
    #     placeholders[0],
    #     # use_bert_score=True,
    #     # compute_mmd=True,
    # )
    # results_list.append(("P1", res_base))
    #
    #
    # for i in range(len(placeholders) - 1):
    #     res = measure_placeholders_drift(
    #         placeholders[i],
    #         placeholders[i + 1],
    #         # use_bert_score=True,
    #         # compute_mmd=True,
    #     )
    #     results_list.append((f"P{i + 1}→P{i + 2}", res))
    #
    # # # Total P1→P5
    # # res_total = measure_placeholders_drift(
    # #     placeholders[0],
    # #     placeholders[-1],
    # #     # use_bert_score=True,
    # #     # compute_mmd=True,
    # # )
    # # results_list.append(("P1→P5", res_total))
    #
    # DriftPlotter.plot_drift_incremental(results_list)
    # print("--- --- --- ---")
