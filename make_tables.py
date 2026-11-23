"""Programmatically generate zero-overfull booktabs/longtable LaTeX tables from subtyping pipeline CSVs."""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd


def _fmt(v, dec=3):
    if pd.isna(v) or not np.isfinite(v):
        return "--"
    if v < 0:
        return f"$-${abs(v):.{dec}f}"
    return f"{v:.{dec}f}"


def _fmt_signed(v, dec=1):
    if pd.isna(v) or not np.isfinite(v):
        return "--"
    if v < 0:
        return f"$-${abs(v):.{dec}f}"
    return f"$+${v:.{dec}f}"


def make_tables(res_dir: str, tab_dir: str) -> None:
    os.makedirs(tab_dir, exist_ok=True)
    df_a = pd.read_csv(os.path.join(res_dir, "trajectory_a_calendar_cutoff.csv"))
    pre = df_a[df_a["cutoff_label"] != "2026_final"]
    fin = df_a[df_a["cutoff_label"] == "2026_final"]

    pmap = {
        "ADNI_P1_clinical_8": "ADNI Clinical (8)",
        "ADNI_P2_clin_fs7_13": "ADNI Clin+FS7 (13)",
        "ADNI_P3_clin_fshist_13": "ADNI Clin+FS Hist (13)",
        "NACC_P1_stable_7": "NACC Stable Core (7)",
        "NACC_P2_uds_switch_naive_9": "NACC Naive Switch (9)",
        "NACC_P3_uds_crosswalk_8": "NACC Crosswalk (8)",
    }
    amap = {"kmeans": "K-means", "gmm_diag": "Diag.\\ GMM", "ward": "Ward"}

    # Table 1: K=3 Main Stability & Clinical Prognosis Summary (All 6 panels x 3 algos = 18 rows)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Twenty-year clustering stability and independent 24-month clinical prognostic separation across ADNI and NACC ($K=3$, 2008--2026).}",
        r"\label{tab:lc02_main_k3}",
        r"\ManuscriptTableSize{7.8}{9.6}",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"Panel (Features) & Algo. & Eval $N$ & Mean ARI$^{\text{a}}$ & Flip$^{\text{a}}$ & Min Step & Sil.$^{\text{b}}$ & CDR $\eta^2$ & MCI RD$^{\text{c}}$ \\",
        r"\midrule",
    ]
    p_order = list(pmap.keys())
    for pi, pid in enumerate(p_order):
        if pi == 3:
            lines.append(r"\midrule")
        for ai, al in enumerate(("kmeans", "gmm_diag", "ward")):
            s_pre = pre[(pre["panel_id"] == pid) & (pre["algo"] == al) & (pre["k"] == 3)]
            s_all = df_a[(df_a["panel_id"] == pid) & (df_a["algo"] == al) & (df_a["k"] == 3)]
            s_fin = fin[(fin["panel_id"] == pid) & (fin["algo"] == al) & (fin["k"] == 3)].iloc[0]
            min_st = s_all["step_ari"].min()
            p_str = pmap[pid] if ai == 0 else ""
            lines.append(
                f"{p_str} & {amap[al]} & {int(s_fin['eval_n']):,} & "
                f"{_fmt(s_pre['ari_to_ref'].mean(), 3)} & {s_pre['flip_rate_to_ref'].mean() * 100.0:.1f}\\% & "
                f"{_fmt(min_st, 3)} & {_fmt(s_fin['silhouette_eval'], 3)} & "
                f"{_fmt(s_fin['out_cdrsb_24m_eta2'], 3)} & {_fmt_signed(s_fin['out_mci_conv_24m_diff'], 1)} \\\\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"\textit{Notes.} $^{\text{a}}$Mean ARI and Hungarian-aligned individual flip rate relative to the 2026 reference solution averaged over annual cutoffs 2008--2025. $^{\text{b}}$Silhouette coefficient on the fixed 30\% evaluation pool at the 2026 reference solution. $^{\text{c}}$Severe-minus-mild cluster 24-month MCI-to-dementia conversion risk difference (percentage points, pp) in the 2026 reference solution.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    with open(os.path.join(tab_dir, "table1_stability_prognosis.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Table 2: NACC Autopsy Neuropathology Gold-Standard Validity (K=3, 3 panels x 3 algos)
    lines2 = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{External autopsy neuropathology validity versus 24-month clinical progression separation in the independent NACC evaluation pool ($K=3$, 2026 reference solution; $n=1{,}166$ ADNC, $n=1{,}939$ Braak, $n=1{,}188$ Thal, $n=1{,}985$ Lewy).}",
        r"\label{tab:lc02_neuropathology}",
        r"\ManuscriptTableSize{7.8}{9.6}",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"NACC Panel & Algo. & CDR $\eta^2$ & ADNC $\eta^2$ & ADNC$\ge$2 & Braak V--VI & Thal 4--5 & Lewy & Flip \\",
        r"\midrule",
    ]
    for pi, pid in enumerate(["NACC_P1_stable_7", "NACC_P2_uds_switch_naive_9", "NACC_P3_uds_crosswalk_8"]):
        if pi > 0:
            lines2.append(r"\midrule")
        for ai, al in enumerate(("kmeans", "gmm_diag", "ward")):
            s_fin = fin[(fin["panel_id"] == pid) & (fin["algo"] == al) & (fin["k"] == 3)].iloc[0]
            s_pre = pre[(pre["panel_id"] == pid) & (pre["algo"] == al) & (pre["k"] == 3)]
            p_str = pmap[pid] if ai == 0 else ""
            lines2.append(
                f"{p_str} & {amap[al]} & {_fmt(s_fin['out_cdrsb_24m_eta2'], 3)} & "
                f"{_fmt(s_fin['np_adnc_ord_eta2'], 3)} & {_fmt_signed(s_fin['np_adnc_int_high_diff'], 1)} & "
                f"{_fmt_signed(s_fin['np_braak_v_vi_diff'], 1)} & {_fmt_signed(s_fin['np_thal_4_5_diff'], 1)} & "
                f"{_fmt_signed(s_fin['np_lewy_any_diff'], 1)} & {s_pre['flip_rate_to_ref'].mean() * 100.0:.1f}\\% \\\\"
            )
    lines2 += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"\textit{Notes.} RD, severe-minus-mild cluster risk difference in percentage points (pp). ADNC $\eta^2$ denotes variance explained across ordinal NPADNC (0--3). Mean Flip denotes 2008--2025 average individual reassignment rate.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    with open(os.path.join(tab_dir, "table2_neuropathology.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines2) + "\n")

    # Table S1: Complete 54-strategy longtable
    lines_s1 = [
        r"{\ManuscriptTableSize{7.8}{9.6}\setlength{\tabcolsep}{2.6pt}",
        r"\begin{longtable}{@{}llrrrrrrr@{}}",
        r"\caption{Complete 54-strategy stability, silhouette, and external validity summary across ADNI and NACC ($K \in \{2,3,4\}$).}\label{tab:lc02_s1_all54}\\",
        r"\toprule",
        r"Panel & Algo. & $K$ & Mean ARI & Flip & Min Step & Sil. & CDR $\eta^2$ & ADNC$\ge$2 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Panel & Algo. & $K$ & Mean ARI & Flip & Min Step & Sil. & CDR $\eta^2$ & ADNC$\ge$2 \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for pid in p_order:
        for al in ("kmeans", "gmm_diag", "ward"):
            for k in (2, 3, 4):
                s_pre = pre[(pre["panel_id"] == pid) & (pre["algo"] == al) & (pre["k"] == k)]
                s_all = df_a[(df_a["panel_id"] == pid) & (df_a["algo"] == al) & (df_a["k"] == k)]
                s_fin = fin[(fin["panel_id"] == pid) & (fin["algo"] == al) & (fin["k"] == k)].iloc[0]
                adnc_s = _fmt_signed(s_fin["np_adnc_int_high_diff"], 1) if pid.startswith("NACC") else "--"
                lines_s1.append(
                    f"{pmap[pid]} & {amap[al]} & {k} & {_fmt(s_pre['ari_to_ref'].mean(), 3)} & "
                    f"{s_pre['flip_rate_to_ref'].mean() * 100.0:.1f}\\% & {_fmt(s_all['step_ari'].min(), 3)} & "
                    f"{_fmt(s_fin['silhouette_eval'], 3)} & {_fmt(s_fin['out_cdrsb_24m_eta2'], 3)} & {adnc_s} \\\\"
                )
    lines_s1.append(r"\end{longtable}}")
    with open(os.path.join(tab_dir, "tableS1_all54.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_s1) + "\n")

    print(f"Generated 3 subtyping LaTeX tables in {tab_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir", default="outputs/subtyping_full")
    parser.add_argument("--tab-dir", default="outputs/tables")
    args = parser.parse_args()
    make_tables(args.res_dir, args.tab_dir)


if __name__ == "__main__":
    main()
