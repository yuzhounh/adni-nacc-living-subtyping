"""Pure synthetic unit tests for the unsupervised subtyping and crosswalk engine (zero private data required)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from subtyping_engine import (
    MONSELL_2016_MOCA_TO_MMSE,
    align_to_reference_hungarian,
    apply_monsell_moca_to_mmse,
    canonicalize_reference_clusters,
    compute_external_validity,
    compute_stability_metrics,
    exact_weighted_ward_clusters,
    fit_and_predict_clusterer,
)


def test_monsell_crosswalk_monotonic_and_exact():
    keys = list(range(31))
    vals = [MONSELL_2016_MOCA_TO_MMSE[k] for k in keys]
    assert all(vals[i] <= vals[i + 1] for i in range(30)), "Crosswalk must be non-decreasing"
    assert MONSELL_2016_MOCA_TO_MMSE[20] == 26
    assert MONSELL_2016_MOCA_TO_MMSE[30] == 30
    assert MONSELL_2016_MOCA_TO_MMSE[0] == 6
    # Test education adjustment (+1 for educ<=12 when moca<30)
    moca = np.array([19.0, 20.0, 30.0, np.nan])
    educ = np.array([10.0, 16.0, 10.0, 12.0])
    eq = apply_monsell_moca_to_mmse(moca, educ, already_adj=False)
    assert eq[0] == 26.0  # 19 + 1 = 20 -> 26
    assert eq[1] == 26.0  # 20 -> 26
    assert eq[2] == 30.0  # capped at 30 -> 30
    assert np.isnan(eq[3])


def test_exact_weighted_ward_matches_sklearn():
    rng = np.random.default_rng(77)
    X = np.vstack([
        rng.normal([0.0, 0.0, 0.0], 0.5, (40, 3)),
        rng.normal([3.0, 2.0, -1.0], 0.6, (45, 3)),
        rng.normal([-2.5, 3.0, 2.0], 0.5, (35, 3)),
    ])
    lbl_sk = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
    lbl_ours = exact_weighted_ward_clusters(X, np.ones(len(X)), n_clusters=3)
    ari = adjusted_rand_score(lbl_sk, lbl_ours)
    assert ari == 1.0, f"Exact weighted Ward should match sklearn Ward with ARI=1.0, got {ari}"


def test_hungarian_alignment_zero_drift_on_identical_fit():
    rng = np.random.default_rng(123)
    X_tr = np.vstack([
        rng.normal([-2, -2, -2, -2], 0.6, (100, 4)),
        rng.normal([0, 0, 0, 0], 0.6, (100, 4)),
        rng.normal([2.5, 2.5, 2.5, 2.5], 0.6, (100, 4)),
    ])
    X_ev = np.vstack([
        rng.normal([-2, -2, -2, -2], 0.6, (50, 4)),
        rng.normal([0, 0, 0, 0], 0.6, (50, 4)),
        rng.normal([2.5, 2.5, 2.5, 2.5], 0.6, (50, 4)),
    ])
    for algo in ("kmeans", "ward", "gmm_diag"):
        fit = fit_and_predict_clusterer(X_tr, X_ev, algo=algo, k=3, seed=42, n_init=10)
        assert not fit.degenerate
        ref_cent, ref_tr, ref_ev = canonicalize_reference_clusters(fit, [1, 1, 1, 1])
        al_tr, al_ev, cdist = align_to_reference_hungarian(fit, ref_cent, fit.mean_x, fit.sd_x)
        assert cdist < 1e-12
        assert np.array_equal(al_ev, ref_ev)
        X_ev_std = (X_ev - fit.mean_x) / fit.sd_x
        st = compute_stability_metrics(al_ev, ref_ev, X_ev_std, k=3)
        assert st["ari_to_ref"] == 1.0
        assert st["flip_rate_to_ref"] == 0.0


def test_external_validity_contrast():
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
    y_bin = np.array([0] * 27 + [1] * 3 + [0] * 15 + [1] * 15 + [0] * 3 + [1] * 27, dtype=float)
    ev = compute_external_validity(labels, k=3, outcome_values=y_bin, is_binary=True)
    assert abs(ev["hi_lo_diff"] - 80.0) < 1e-9  # 90% - 10% = 80 pp
    assert ev["eta2"] > 0.4


def run_all() -> None:
    tests = [
        test_monsell_crosswalk_monotonic_and_exact,
        test_exact_weighted_ward_matches_sklearn,
        test_hungarian_alignment_zero_drift_on_identical_fit,
        test_external_validity_contrast,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"All {len(tests)}/{len(tests)} subtyping synthetic unit tests passed.")


if __name__ == "__main__":
    run_all()
