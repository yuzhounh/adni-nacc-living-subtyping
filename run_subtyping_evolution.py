"""End-to-end dual-cohort unsupervised subtyping stability, prognosis dissociation, and neuropathology validity pipeline."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from build_subtyping_cohorts import (
    PANEL_SPECS,
    SubtypingCohortBundle,
    build_adni_subtyping_cohort,
    build_nacc_subtyping_cohort,
)
from subtyping_engine import (
    align_to_reference_hungarian,
    canonicalize_reference_clusters,
    compute_external_validity,
    compute_stability_metrics,
    fit_and_predict_clusterer,
    transform_with_preprocessor,
)


ANNUAL_CUTOFF_YEARS = list(range(2008, 2026))
ALGOS = ("kmeans", "ward", "gmm_diag")
KS = (2, 3, 4)


def make_doubling_grid(n_full: int, min_target: int) -> List[int]:
    sizes = [n_full]
    k = 1
    while True:
        val = int(np.floor(n_full / (2 ** k) + 0.5))
        if val < max(40, int(min_target * 0.7)):
            break
        if val not in sizes:
            sizes.append(val)
        k += 1
    return sorted(sizes)


def _filter_panel_eligible(df: pd.DataFrame, feat_cols: List[str], min_frac: float = 0.75) -> pd.DataFrame:
    """Keep subjects with >= min_frac non-missing features in the panel.
    For UDS switch panel (NACC_P2_uds_switch_naive_9), MMSE & MoCA are mutually exclusive across UDS2/3,
    so 8/9 = 88.9% >= 75% still keeps both UDS2 and UDS3 subjects naturally!
    """
    avail = df[feat_cols].notna().mean(axis=1)
    return df[avail >= min_frac].copy().reset_index(drop=True)


def run_strategy_worker(payload):
    """Run Trajectory A (19 calendar cutoffs) and Trajectory B (chronological doubling) for one (panel, algo, K) strategy."""
    cohort, panel_id, algo, k, df_elig, external_outcomes, final_cutoff, n_reps_b, seed = payload
    t0 = time.time()
    feat_cols, sev_signs = PANEL_SPECS[panel_id]

    dev_full = df_elig[~df_elig["is_eval"]].copy().sort_values("subj_id").reset_index(drop=True)
    ev_full = df_elig[df_elig["is_eval"]].copy().sort_values("subj_id").reset_index(drop=True)
    assert len(set(dev_full["subj_id"]).intersection(set(ev_full["subj_id"]))) == 0

    X_dev_full = dev_full[feat_cols].to_numpy(dtype=np.float64)
    X_ev = ev_full[feat_cols].to_numpy(dtype=np.float64)
    ext_arrays = {col: (ev_full[col].to_numpy(dtype=np.float64), is_bin) for col, is_bin in external_outcomes}

    # Fit canonical final 2026 reference solution
    n_init_ref = 30 if algo == "kmeans" else 15
    ref_fit = fit_and_predict_clusterer(X_dev_full, X_ev, algo=algo, k=k, seed=seed, n_init=n_init_ref)
    ref_cent_std, ref_tr_lbl, ref_ev_lbl = canonicalize_reference_clusters(ref_fit, sev_signs)
    X_ev_ref_std = transform_with_preprocessor(X_ev, ref_fit.medians, ref_fit.mean_x, ref_fit.sd_x)

    # Trajectory A: 19 annual calendar cutoffs (2008..2025 + 2026_final)
    cutoffs = [pd.Timestamp(f"{yr}-12-31") for yr in ANNUAL_CUTOFF_YEARS] + [final_cutoff]
    cut_labels = [str(yr) for yr in ANNUAL_CUTOFF_YEARS] + ["2026_final"]

    rows_a: List[Dict] = []
    prev_ev_lbl = None

    for cut_dt, cut_lbl in zip(cutoffs, cut_labels):
        is_final = (cut_lbl == "2026_final")
        sub_tr = dev_full if is_final else dev_full[dev_full["bl_dt"] <= cut_dt].sort_values("subj_id")
        n_tr = len(sub_tr)
        if n_tr < max(30, k * 10):
            continue
        if is_final:
            cur_fit = ref_fit
            al_ev = ref_ev_lbl
            cdist = 0.0
        else:
            X_tr = sub_tr[feat_cols].to_numpy(dtype=np.float64)
            n_init_a = 30 if algo == "kmeans" else 15
            cur_fit = fit_and_predict_clusterer(X_tr, X_ev, algo=algo, k=k, seed=seed, n_init=n_init_a)
            _, al_ev, cdist = align_to_reference_hungarian(cur_fit, ref_cent_std, ref_fit.mean_x, ref_fit.sd_x)

        st = compute_stability_metrics(al_ev, ref_ev_lbl, X_ev_ref_std, k=k)
        step_ari = float(adjusted_rand_score(prev_ev_lbl, al_ev)) if prev_ev_lbl is not None else float("nan")
        prev_ev_lbl = al_ev.copy()

        row = {
            "cohort": cohort,
            "panel_id": panel_id,
            "algo": algo,
            "k": int(k),
            "trajectory": "A_calendar_cutoff",
            "cutoff_label": cut_lbl,
            "cutoff_date": str(cut_dt.date()),
            "train_n": int(n_tr),
            "eval_n": int(len(ev_full)),
            "degenerate": int(cur_fit.degenerate),
            "centroid_dist_to_ref": cdist,
            "step_ari": step_ari,
        }
        row.update(st)
        for col, (arr, is_bin) in ext_arrays.items():
            ev_m = compute_external_validity(al_ev, k=k, outcome_values=arr, is_binary=is_bin)
            row[f"{col}_eta2"] = ev_m["eta2"]
            row[f"{col}_diff"] = ev_m["hi_lo_diff"]
        rows_a.append(row)

    # Trajectory B: chronological doubling on exact bl_dt
    min_target = 60 if cohort == "ADNI" else 200
    sizes = make_doubling_grid(len(dev_full), min_target=min_target)
    dev_dt = dev_full.sort_values(["bl_dt", "subj_id"]).reset_index(drop=True)
    date_vals = dev_dt["bl_dt"].to_numpy()
    X_dt = dev_dt[feat_cols].to_numpy(dtype=np.float64)
    _, first_idx, counts = np.unique(date_vals, return_index=True, return_counts=True)
    tie_groups = [(int(s), int(s + c)) for s, c in zip(first_idx, counts) if c > 1]

    rng = np.random.default_rng(seed + 503)
    rows_b: List[Dict] = []

    for sz in sizes:
        is_full = (sz == len(dev_dt))
        boundary_in_tie = False if is_full else (date_vals[sz - 1] == date_vals[sz])
        # Cap tie reps on large NACC sizes (where permuting 5 same-day ties out of >5000 has negligible variance)
        reps_here = 1 if (is_full or not boundary_in_tie) else (min(n_reps_b, 20) if sz >= 3000 else n_reps_b)

        rep_list = []
        for _ in range(reps_here):
            if is_full:
                cur_fit = ref_fit
                al_ev = ref_ev_lbl
                cdist = 0.0
            else:
                if not boundary_in_tie:
                    idx_sub = np.arange(sz)
                else:
                    perm = np.arange(len(dev_dt))
                    for s_g, e_g in tie_groups:
                        if s_g < sz < e_g:
                            perm[s_g:e_g] = rng.permutation(perm[s_g:e_g])
                    idx_sub = np.sort(perm[:sz])
                sub_ids = dev_dt["subj_id"].iloc[idx_sub].to_numpy()
                idx_ord = idx_sub[np.argsort(sub_ids)]
                n_init_b = 15 if algo == "kmeans" else 8
                cur_fit = fit_and_predict_clusterer(X_dt[idx_ord], X_ev, algo=algo, k=k, seed=seed, n_init=n_init_b)
                _, al_ev, cdist = align_to_reference_hungarian(cur_fit, ref_cent_std, ref_fit.mean_x, ref_fit.sd_x)

            st = compute_stability_metrics(al_ev, ref_ev_lbl, X_ev_ref_std, k=k)
            d_r = {"centroid_dist_to_ref": cdist, "degenerate": int(cur_fit.degenerate)}
            d_r.update(st)
            for col, (arr, is_bin) in ext_arrays.items():
                ev_m = compute_external_validity(al_ev, k=k, outcome_values=arr, is_binary=is_bin)
                d_r[f"{col}_eta2"] = ev_m["eta2"]
                d_r[f"{col}_diff"] = ev_m["hi_lo_diff"]
            rep_list.append(d_r)

        df_r = pd.DataFrame(rep_list)
        row_b = {
            "cohort": cohort,
            "panel_id": panel_id,
            "algo": algo,
            "k": int(k),
            "trajectory": "B_chronological_doubling",
            "train_n": int(sz),
            "eval_n": int(len(ev_full)),
            "n_reps_run": int(reps_here),
            "boundary_tied": int(boundary_in_tie),
            "max_bl_date": str(pd.Timestamp(date_vals[sz - 1]).date()),
        }
        for col in df_r.columns:
            vals = df_r[col].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row_b[col] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
            if col in ("ari_to_ref", "flip_rate_to_ref", "out_mci_conv_24m_diff", "np_adnc_int_high_diff"):
                row_b[f"{col}_p025"] = float(np.percentile(vals, 2.5)) if len(vals) > 0 else float("nan")
                row_b[f"{col}_p975"] = float(np.percentile(vals, 97.5)) if len(vals) > 0 else float("nan")
        rows_b.append(row_b)

    # Zero-drift anchor check: B(N_full) == A(2026_final)
    a_fin = [r for r in rows_a if r["cutoff_label"] == "2026_final"][0]
    b_fin = [r for r in rows_b if r["train_n"] == len(dev_full)][0]
    assert abs(a_fin["ari_to_ref"] - 1.0) < 1e-12
    assert abs(b_fin["ari_to_ref"] - 1.0) < 1e-12
    assert abs(a_fin["flip_rate_to_ref"]) < 1e-12
    assert abs(b_fin["flip_rate_to_ref"]) < 1e-12

    return f"{panel_id}|{algo}|K={k}", rows_a, rows_b, round(time.time() - t0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-Cohort ADNI and NACC Unsupervised Subtyping and Neuropathology Pipeline")
    parser.add_argument("--adni-raw-dir", default=os.environ.get("ADNI_RAW_DIR", "data/raw/adni_route_a"))
    parser.add_argument("--adni-extract-dir", default=os.environ.get("ADNI_EXTRACT_DIR", "data/raw/adni_extract"))
    parser.add_argument("--nacc-parquet", default=os.environ.get("NACC_PARQUET", "data/raw/nacc/investigator_nacc75.parquet"))
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--n-reps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    n_reps = 5 if args.pilot else args.n_reps
    mode_tag = "pilot" if args.pilot else "full"
    print(f"=== Dual-Cohort Subtyping Pipeline ({mode_tag}, n_reps_B={n_reps}, workers={args.workers}) ===", flush=True)

    adni_bundle, adni_meta = build_adni_subtyping_cohort(args.adni_raw_dir, args.adni_extract_dir)
    nacc_bundle, nacc_meta = build_nacc_subtyping_cohort(args.nacc_parquet)
    print(f"Built ADNI (N={adni_meta['n_impaired']}, eval={adni_meta['n_eval']}) & NACC (N={nacc_meta['n_impaired']}, eval={nacc_meta['n_eval']}) in {time.time() - t0:.1f}s.", flush=True)
    print("  NACC eval outcomes:", nacc_meta["outcome_n_eval"], flush=True)
    print("  ADNI eval outcomes:", adni_meta["outcome_n_eval"], flush=True)

    tasks = []
    panel_counts = {}
    idx = 0
    for bundle in (adni_bundle, nacc_bundle):
        for pid in bundle.panel_ids:
            fcols, _ = PANEL_SPECS[pid]
            df_el = _filter_panel_eligible(bundle.df, fcols, min_frac=0.75)
            panel_counts[pid] = {
                "total": int(len(df_el)),
                "dev": int((~df_el["is_eval"]).sum()),
                "eval": int(df_el["is_eval"].sum()),
            }
            for algo in ALGOS:
                for k in KS:
                    tasks.append((
                        bundle.cohort, pid, algo, k, df_el, bundle.external_outcomes,
                        bundle.final_cutoff, n_reps, 20261123 + idx * 13
                    ))
                    idx += 1

    print(f"Panel eligible counts: {json.dumps(panel_counts)}", flush=True)
    print(f"Launching {len(tasks)} strategies across {args.workers} workers...", flush=True)

    all_a, all_b = [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
        for tag, ra, rb, dt_s in ex.map(run_strategy_worker, tasks):
            all_a.extend(ra)
            all_b.extend(rb)
            print(f"  [DONE] {tag:38s} in {dt_s:5.1f}s (A={len(ra)}, B={len(rb)})", flush=True)

    sub_dir = os.path.join(args.out_dir, f"subtyping_{mode_tag}")
    os.makedirs(sub_dir, exist_ok=True)
    pd.DataFrame(all_a).to_csv(os.path.join(sub_dir, "trajectory_a_calendar_cutoff.csv"), index=False)
    pd.DataFrame(all_b).to_csv(os.path.join(sub_dir, "trajectory_b_chronological_doubling.csv"), index=False)

    meta_out = {
        "mode": mode_tag,
        "n_reps_b": n_reps,
        "elapsed_seconds": round(time.time() - t0, 2),
        "adni_meta": adni_meta,
        "nacc_meta": nacc_meta,
        "panel_counts": panel_counts,
        "n_strategies": len(tasks),
    }
    with open(os.path.join(sub_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    print(f"\n=== Subtyping Pipeline ({mode_tag}) FINISHED in {time.time() - t0:.1f}s -> {sub_dir} ===", flush=True)


if __name__ == "__main__":
    main()
