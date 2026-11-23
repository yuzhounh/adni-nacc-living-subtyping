"""Independent unsupervised subtyping engine with exact weighted Ward, Hungarian alignment, and Monsell (2016) crosswalk for dual-cohort ADNI/NACC unsupervised subtyping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture


# Official Monsell et al. (2016) NACC UDS Crosswalk Study Table 3 (PMC4834278):
# Education-adjusted MoCA (0..30) -> Equivalent MMSE (0..30)
MONSELL_2016_MOCA_TO_MMSE: Dict[int, int] = {
    0: 6, 1: 9, 2: 10, 3: 11, 4: 12, 5: 12, 6: 13, 7: 14, 8: 15, 9: 15,
    10: 16, 11: 17, 12: 18, 13: 19, 14: 20, 15: 21, 16: 22, 17: 23, 18: 24, 19: 25,
    20: 26, 21: 27, 22: 28, 23: 28, 24: 29, 25: 29, 26: 29, 27: 30, 28: 30, 29: 30, 30: 30,
}


def apply_monsell_moca_to_mmse(moca: np.ndarray, educ: Optional[np.ndarray] = None, already_adj: bool = False) -> np.ndarray:
    """Convert MoCA scores to MMSE-equivalent scores using Monsell et al. (2016) Table 3 (zero leakage)."""
    arr = np.asarray(moca, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(arr) & (arr >= 0) & (arr <= 30)
    if not np.any(valid):
        return out
    adj = arr.copy()
    if (not already_adj) and (educ is not None):
        ed = np.asarray(educ, dtype=np.float64)
        add_pt = valid & np.isfinite(ed) & (ed <= 12.0) & (adj < 30.0)
        adj[add_pt] = np.minimum(adj[add_pt] + 1.0, 30.0)
    ints = np.rint(adj[valid]).astype(np.int64)
    ints = np.clip(ints, 0, 30)
    out[valid] = np.array([MONSELL_2016_MOCA_TO_MMSE[int(v)] for v in ints], dtype=np.float64)
    return out


def fit_train_preprocessor(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit median imputation and z-score standardization strictly inside training data."""
    X = np.asarray(X_train, dtype=np.float64)
    p = X.shape[1]
    medians = np.zeros(p, dtype=np.float64)
    X_imp = X.copy()
    for j in range(p):
        col = X[:, j]
        ok = np.isfinite(col)
        med = float(np.median(col[ok])) if np.any(ok) else 0.0
        medians[j] = med
        if not np.all(ok):
            X_imp[~ok, j] = med
    mean_x = X_imp.mean(axis=0)
    sd_x = X_imp.std(axis=0, ddof=0)
    sd_safe = np.where(sd_x > 1e-12, sd_x, 1.0)
    return (X_imp - mean_x) / sd_safe, medians, mean_x, sd_safe


def transform_with_preprocessor(
    X: np.ndarray, medians: np.ndarray, mean_x: np.ndarray, sd_safe: np.ndarray
) -> np.ndarray:
    X_imp = np.asarray(X, dtype=np.float64).copy()
    for j in range(X_imp.shape[1]):
        bad = ~np.isfinite(X_imp[:, j])
        if np.any(bad):
            X_imp[bad, j] = medians[j]
    return (X_imp - mean_x) / sd_safe


def exact_weighted_ward_clusters(
    centroids: np.ndarray, weights: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Exact agglomerative Ward clustering on weighted centroids minimizing delta_SSE = (w_a*w_b)/(w_a+w_b)*||mu_a-mu_b||^2."""
    m = len(centroids)
    if m <= n_clusters:
        return np.arange(m, dtype=np.int64) % n_clusters

    cur_mu = centroids.astype(np.float64).copy()
    cur_w = weights.astype(np.float64).copy()
    members: List[List[int]] = [[i] for i in range(m)]
    active = np.ones(m, dtype=bool)

    # Initial pairwise Ward merge cost matrix (symmetric, inf on diagonal)
    diff = cur_mu[:, None, :] - cur_mu[None, :, :]
    sq_dist = np.einsum("ijk,ijk->ij", diff, diff)
    w_factor = (cur_w[:, None] * cur_w[None, :]) / (cur_w[:, None] + cur_w[None, :])
    cost = w_factor * sq_dist
    np.fill_diagonal(cost, np.inf)

    n_active = m
    while n_active > n_clusters:
        flat_idx = int(np.argmin(cost))
        a, b = divmod(flat_idx, m)
        if a > b:
            a, b = b, a
        # Merge b into a
        wa, wb = cur_w[a], cur_w[b]
        w_new = wa + wb
        cur_mu[a] = (wa * cur_mu[a] + wb * cur_mu[b]) / w_new
        cur_w[a] = w_new
        members[a].extend(members[b])
        members[b] = []
        active[b] = False

        # Invalidate b row/col
        cost[b, :] = np.inf
        cost[:, b] = np.inf

        # Update distances from new a to all other active nodes
        act_idx = np.flatnonzero(active)
        act_idx = act_idx[act_idx != a]
        d_vec = cur_mu[act_idx] - cur_mu[a]
        sq = np.einsum("ij,ij->i", d_vec, d_vec)
        wf = (w_new * cur_w[act_idx]) / (w_new + cur_w[act_idx])
        new_c = wf * sq
        cost[a, act_idx] = new_c
        cost[act_idx, a] = new_c
        cost[a, a] = np.inf
        n_active -= 1

    labels = np.empty(m, dtype=np.int64)
    cid = 0
    for i in range(m):
        if active[i]:
            for orig_idx in members[i]:
                labels[orig_idx] = cid
            cid += 1
    return labels


@dataclass
class ClusterFitResult:
    train_labels: np.ndarray
    eval_labels: np.ndarray
    centroids_raw: np.ndarray   # (K, p) in original feature scale
    medians: np.ndarray
    mean_x: np.ndarray
    sd_x: np.ndarray
    degenerate: bool


def fit_and_predict_clusterer(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    algo: str,
    k: int,
    seed: int = 20261123,
    n_init: int = 30,
    ward_micro_threshold: int = 3500,
    ward_micro_k: int = 400,
) -> ClusterFitResult:
    """Fit clustering algorithm strictly inside X_train and map X_eval without label leakage."""
    X_tr_std, medians, mean_x, sd_safe = fit_train_preprocessor(X_train)
    X_ev_std = transform_with_preprocessor(X_eval, medians, mean_x, sd_safe)
    n_tr, p = X_tr_std.shape

    if algo == "kmeans":
        km = KMeans(n_clusters=k, n_init=n_init, tol=1e-5, random_state=seed)
        tr_lbl = km.fit_predict(X_tr_std)
        ev_lbl = km.predict(X_ev_std)
        cent_std = km.cluster_centers_
    elif algo == "ward":
        if n_tr <= ward_micro_threshold:
            ward = AgglomerativeClustering(n_clusters=k, linkage="ward")
            tr_lbl = ward.fit_predict(X_tr_std)
        else:
            mk = min(ward_micro_k, n_tr)
            km_micro = KMeans(n_clusters=mk, n_init=5, random_state=seed)
            micro_lbl = km_micro.fit_predict(X_tr_std)
            counts = np.bincount(micro_lbl, minlength=mk).astype(np.float64)
            non_empty = counts > 0
            c_mu = km_micro.cluster_centers_[non_empty]
            c_w = counts[non_empty]
            macro_of_ne = exact_weighted_ward_clusters(c_mu, c_w, n_clusters=k)
            macro_map = np.zeros(mk, dtype=np.int64)
            macro_map[non_empty] = macro_of_ne
            tr_lbl = macro_map[micro_lbl]

        cent_std = np.zeros((k, p), dtype=np.float64)
        for c in range(k):
            mask = tr_lbl == c
            if np.any(mask):
                cent_std[c] = X_tr_std[mask].mean(axis=0)
        # Nearest training centroid mapping for eval set (zero leakage)
        dists = np.sum((X_ev_std[:, None, :] - cent_std[None, :, :]) ** 2, axis=2)
        ev_lbl = np.argmin(dists, axis=1).astype(np.int64)
    elif algo == "gmm_diag":
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=1e-4,
            tol=1e-4,
            n_init=n_init,
            max_iter=300,
            random_state=seed,
        )
        gmm.fit(X_tr_std)
        tr_lbl = gmm.predict(X_tr_std).astype(np.int64)
        ev_lbl = gmm.predict(X_ev_std).astype(np.int64)
        cent_std = gmm.means_
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    uniq_tr = len(np.unique(tr_lbl))
    degenerate = (uniq_tr < k)
    cent_raw = cent_std * sd_safe[None, :] + mean_x[None, :]
    return ClusterFitResult(
        train_labels=tr_lbl,
        eval_labels=ev_lbl,
        centroids_raw=cent_raw,
        medians=medians,
        mean_x=mean_x,
        sd_x=sd_safe,
        degenerate=degenerate,
    )


def canonicalize_reference_clusters(
    ref_fit: ClusterFitResult, severity_signs: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order reference clusters 0 (mildest) .. K-1 (most severe) by training-centroid composite severity index."""
    signs = np.asarray(severity_signs, dtype=np.float64)
    cent_ref_std = (ref_fit.centroids_raw - ref_fit.mean_x[None, :]) / ref_fit.sd_x[None, :]
    sev_scores = (cent_ref_std * signs[None, :]).mean(axis=1)
    order = np.argsort(sev_scores, kind="mergesort")
    perm = np.empty_like(order)
    perm[order] = np.arange(len(order), dtype=np.int64)

    new_cent_ref_std = cent_ref_std[order]
    new_eval_lbl = perm[ref_fit.eval_labels]
    new_train_lbl = perm[ref_fit.train_labels]
    return new_cent_ref_std, new_train_lbl, new_eval_lbl


def align_to_reference_hungarian(
    cur_fit: ClusterFitResult,
    ref_cent_std: np.ndarray,
    ref_mean: np.ndarray,
    ref_sd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Leakage-free Hungarian alignment of training centroids projected into fixed reference standardized space."""
    cur_cent_in_ref = (cur_fit.centroids_raw - ref_mean[None, :]) / ref_sd[None, :]
    diff = cur_cent_in_ref[:, None, :] - ref_cent_std[None, :, :]
    cost_mat = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    perm = np.empty(len(row_ind), dtype=np.int64)
    perm[row_ind] = col_ind
    mean_centroid_dist = float(cost_mat[row_ind, col_ind].mean())
    aligned_eval = perm[cur_fit.eval_labels]
    aligned_train = perm[cur_fit.train_labels]
    return aligned_train, aligned_eval, mean_centroid_dist


def compute_external_validity(
    aligned_eval_labels: np.ndarray, k: int, outcome_values: np.ndarray, is_binary: bool = False
) -> Dict[str, float]:
    """Compute permutation-invariant eta^2 and severe-minus-mild (K-1 minus 0) contrast on external outcomes."""
    y = np.asarray(outcome_values, dtype=np.float64)
    ok = np.isfinite(y)
    if int(np.sum(ok)) < 20:
        return {"n": int(np.sum(ok)), "eta2": float("nan"), "hi_lo_diff": float("nan")}
    yv = y[ok]
    lv = aligned_eval_labels[ok]
    grand_mean = float(np.mean(yv))
    ss_tot = float(np.sum((yv - grand_mean) ** 2))
    if ss_tot <= 1e-12:
        return {"n": int(len(yv)), "eta2": 0.0, "hi_lo_diff": 0.0}

    ss_bet = 0.0
    means = {}
    for c in range(k):
        m = lv == c
        cnt = int(np.sum(m))
        if cnt > 0:
            mc = float(np.mean(yv[m]))
            means[c] = mc
            ss_bet += cnt * ((mc - grand_mean) ** 2)
    eta2 = float(ss_bet / ss_tot)
    if (0 in means) and ((k - 1) in means):
        diff = (means[k - 1] - means[0]) * (100.0 if is_binary else 1.0)
    else:
        diff = float("nan")
    return {"n": int(len(yv)), "eta2": eta2, "hi_lo_diff": float(diff)}


def compute_stability_metrics(
    aligned_eval: np.ndarray,
    ref_eval: np.ndarray,
    X_eval_std_ref: np.ndarray,
    k: int,
) -> Dict[str, float]:
    """Compute ARI, NMI, flip rate, cluster balance, and silhouette on fixed evaluation set."""
    ari = float(adjusted_rand_score(ref_eval, aligned_eval))
    nmi = float(normalized_mutual_info_score(ref_eval, aligned_eval))
    flip = float(np.mean(aligned_eval != ref_eval))
    counts = np.bincount(aligned_eval, minlength=k)
    shares = counts / max(len(aligned_eval), 1)
    if len(np.unique(aligned_eval)) >= 2:
        sample_sz = 3000 if len(aligned_eval) > 3500 else None
        sil = float(silhouette_score(X_eval_std_ref, aligned_eval, sample_size=sample_sz, random_state=42))
    else:
        sil = float("nan")
    return {
        "ari_to_ref": ari,
        "nmi_to_ref": nmi,
        "flip_rate_to_ref": flip,
        "min_cluster_share": float(np.min(shares)),
        "max_cluster_share": float(np.max(shares)),
        "silhouette_eval": sil,
    }
