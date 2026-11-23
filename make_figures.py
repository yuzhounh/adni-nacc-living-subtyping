"""Generate 6 publication-grade PDF/PNG figures from subtyping pipeline CSVs."""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "coral": "#D95F02",
    "purple": "#7570B3",
    "gold": "#E6AB02",
    "crimson": "#B2182B",
    "slate": "#4D4D4D",
    "blue": "#2166AC",
    "green": "#1B7837",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.3,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
    })


def _save(fig: plt.Figure, out_dir: str, stem: str) -> None:
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _yr_col(df: pd.DataFrame) -> pd.Series:
    return df["cutoff_label"].replace("2026_final", "2026").astype(int)


def make_fig1(res_dir: str, fig_dir: str) -> None:
    """Figure 1: NACC UDS 2->3 hard battery switch collapse vs Monsell (2016) crosswalk rescue."""
    df_a = pd.read_csv(os.path.join(res_dir, "trajectory_a_calendar_cutoff.csv"))
    df_a["yr"] = _yr_col(df_a)
    nacc_k3 = df_a[(df_a["cohort"] == "NACC") & (df_a["k"] == 3)]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8))
    p_styles = [
        ("NACC_P1_stable_7", PALETTE["navy"], "-", "o", "Stable 7-feature core (no switch)"),
        ("NACC_P2_uds_switch_naive_9", PALETTE["crimson"], "--", "s", "Naive UDS switch (MMSE + MoCA)"),
        ("NACC_P3_uds_crosswalk_8", PALETTE["teal"], "-", "D", "Monsell (2016) Crosswalk (8 feat.)"),
    ]

    # Panel A: GMM ARI_to_ref over 20 years
    ax = axes[0, 0]
    for pid, clr, ls, mk, lbl in p_styles:
        s = nacc_k3[(nacc_k3["panel_id"] == pid) & (nacc_k3["algo"] == "gmm_diag")].sort_values("yr")
        ax.plot(s["yr"], s["ari_to_ref"], color=clr, ls=ls, marker=mk, ms=4.2, lw=2.1, label=lbl)
    ax.axvline(2015, color=PALETTE["blue"], ls="-.", lw=1.0, alpha=0.65)
    ax.text(2015.25, 0.10, "UDS 3.0\n(2015)", fontsize=7.8, color=PALETTE["blue"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))
    ax.set_xticks([2008, 2011, 2014, 2017, 2020, 2023, 2026])
    ax.set_ylim(0.05, 1.06)
    ax.set_title("A. Diagonal GMM Stability to 2026 Reference (K=3)")
    ax.set_xlabel("Calendar Cutoff Year")
    ax.set_ylabel("ARI vs. 2026 Reference")

    # Panel B: GMM Step-to-Step ARI
    ax = axes[0, 1]
    for pid, clr, ls, mk, lbl in p_styles:
        s = nacc_k3[(nacc_k3["panel_id"] == pid) & (nacc_k3["algo"] == "gmm_diag")].dropna(subset=["step_ari"]).sort_values("yr")
        ax.plot(s["yr"], s["step_ari"], color=clr, ls=ls, marker=mk, ms=4.2, lw=2.1, label=lbl)
    ax.axvline(2015, color=PALETTE["blue"], ls="-.", lw=1.0, alpha=0.65)
    ax.annotate("UDS3 basin collapse\n(Step ARI = 0.156)", xy=(2015, 0.156), xytext=(2015.8, 0.28),
                arrowprops=dict(arrowstyle="->", color=PALETTE["crimson"], lw=1.2),
                fontsize=7.8, color=PALETTE["crimson"], fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
    ax.set_xticks([2009, 2012, 2015, 2018, 2021, 2024, 2026])
    ax.set_ylim(0.05, 1.06)
    ax.set_title("B. Adjacent-Year Step ARI (Diagonal GMM, K=3)")
    ax.set_xlabel("Calendar Cutoff Year")
    ax.set_ylabel("Step-to-Step ARI")

    # Panel C: Mean Individual Flip Rate (2008-2025) by Algorithm x Panel
    ax = axes[1, 0]
    pre = nacc_k3[nacc_k3["cutoff_label"] != "2026_final"]
    algos = ["kmeans", "gmm_diag", "ward"]
    albls = ["K-means", "Diagonal GMM", "Ward (Micro-Weighted)"]
    x = np.arange(len(algos))
    w = 0.25
    for pi, (pid, clr, _, _, lbl) in enumerate(p_styles):
        vals = [pre[(pre["panel_id"] == pid) & (pre["algo"] == al)]["flip_rate_to_ref"].mean() * 100.0 for al in algos]
        bars = ax.bar(x + (pi - 1) * w, vals, width=w, color=clr, label=lbl)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.9, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(albls)
    ax.set_ylim(0, 48)
    ax.set_title("C. Mean 2008-2025 Individual Flip Rate (K=3)")
    ax.set_ylabel("Mean Flip Rate vs. 2026 Reference (%)")

    # Panel D: GMM Individual Flip Rate Trajectory
    ax = axes[1, 1]
    max_fl = 50.0
    for pid, clr, ls, mk, lbl in p_styles:
        s = nacc_k3[(nacc_k3["panel_id"] == pid) & (nacc_k3["algo"] == "gmm_diag")].sort_values("yr")
        fl = s["flip_rate_to_ref"] * 100.0
        max_fl = max(max_fl, float(fl.max()))
        ax.plot(s["yr"], fl, color=clr, ls=ls, marker=mk, ms=4.2, lw=2.1, label=lbl)
    ax.axvline(2015, color=PALETTE["blue"], ls="-.", lw=1.0, alpha=0.65)
    ax.set_xticks([2008, 2011, 2014, 2017, 2020, 2023, 2026])
    ax.set_ylim(-2.5, max_fl + 7.0)
    ax.set_title("D. Diagonal GMM Individual Flip Rate Trajectory (K=3)")
    ax.set_xlabel("Calendar Cutoff Year")
    ax.set_ylabel("Eval Participants Reassigned (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, fig_dir, "fig1_nacc_uds_switch_collapse_rescue")


def make_fig2(res_dir: str, fig_dir: str) -> None:
    """Figure 2: Dual Dissociation - Strong 24m Clinical Prognosis vs Weak Autopsy ADNC & Unstable Clustering."""
    df_a = pd.read_csv(os.path.join(res_dir, "trajectory_a_calendar_cutoff.csv"))
    df_a["yr"] = _yr_col(df_a)
    nacc_k3 = df_a[(df_a["cohort"] == "NACC") & (df_a["k"] == 3)]
    pre = nacc_k3[nacc_k3["cutoff_label"] != "2026_final"]
    fin = nacc_k3[nacc_k3["cutoff_label"] == "2026_final"]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))

    # Panel A: Mean Stability (Flip Rate) vs Mean 24m CDR-SB eta^2 with non-overlapping leader annotations
    ax = axes[0]
    p_map = {
        "NACC_P1_stable_7": ("Stable 7", PALETTE["navy"]),
        "NACC_P2_uds_switch_naive_9": ("Naive Switch", PALETTE["crimson"]),
        "NACC_P3_uds_crosswalk_8": ("Crosswalk", PALETTE["teal"]),
    }
    a_mk = {"kmeans": ("o", "KM"), "gmm_diag": ("s", "GMM"), "ward": ("^", "Ward")}
    # Exact non-overlapping absolute label coordinates (tx, ty)
    label_pos = {
        ("NACC_P3_uds_crosswalk_8", "gmm_diag"): (10.5, 0.580),
        ("NACC_P3_uds_crosswalk_8", "kmeans"): (10.5, 0.556),
        ("NACC_P1_stable_7", "gmm_diag"): (22.5, 0.562),
        ("NACC_P3_uds_crosswalk_8", "ward"): (27.5, 0.532),
        ("NACC_P2_uds_switch_naive_9", "ward"): (25.5, 0.482),
        ("NACC_P1_stable_7", "ward"): (15.0, 0.448),
        ("NACC_P2_uds_switch_naive_9", "kmeans"): (1.5, 0.448),
        ("NACC_P1_stable_7", "kmeans"): (1.5, 0.415),
        ("NACC_P2_uds_switch_naive_9", "gmm_diag"): (24.0, 0.428),
    }
    for pid, (plbl, clr) in p_map.items():
        for al, (mk, albl) in a_mk.items():
            s = pre[(pre["panel_id"] == pid) & (pre["algo"] == al)]
            x_v = float(s["flip_rate_to_ref"].mean() * 100.0)
            y_v = float(s["out_cdrsb_24m_eta2"].mean())
            ax.scatter(x_v, y_v, color=clr, marker=mk, s=85, edgecolor="white", linewidth=0.8, zorder=5)
            tx, ty = label_pos[(pid, al)]
            ax.annotate(
                f"{plbl} ({albl})",
                xy=(x_v, y_v),
                xytext=(tx, ty),
                fontsize=7.2,
                color=clr,
                fontweight="normal",
                arrowprops=dict(arrowstyle="-", color=clr, lw=0.75, alpha=0.65),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.6),
                zorder=6,
            )
    ax.set_xlim(0, 47)
    ax.set_ylim(0.36, 0.60)
    ax.set_xlabel("Mean 2008-2025 Individual Flip Rate (%)")
    ax.set_ylabel("Mean 24m CDR-SB Variance Explained ($\\eta^2$)")
    ax.set_title("A. Stability-Prognosis Dissociation (NACC, K=3)")

    # Panel B: 20-year trajectory of Clinical 24m CDR-SB eta^2 vs Autopsy ADNC eta^2 (Crosswalk K-means & GMM)
    ax = axes[1]
    for al, clr, ls, lbl in [
        ("kmeans", PALETTE["navy"], "-", "K-means"),
        ("gmm_diag", PALETTE["teal"], "--", "Diagonal GMM"),
        ("ward", PALETTE["coral"], ":", "Ward"),
    ]:
        s = nacc_k3[(nacc_k3["panel_id"] == "NACC_P3_uds_crosswalk_8") & (nacc_k3["algo"] == al)].sort_values("yr")
        ax.plot(s["yr"], s["out_cdrsb_24m_eta2"], color=clr, ls=ls, lw=2.0, label=f"{lbl}: 24m CDR-SB $\\eta^2$")
        ax.plot(s["yr"], s["np_adnc_ord_eta2"], color=clr, ls=ls, lw=2.0, marker="x", ms=4, alpha=0.75)
    ax.text(2010, 0.44, "24m Clinical CDR-SB $\\eta^2$\n(0.49 - 0.56)", fontsize=8.3, color=PALETTE["navy"], fontweight="bold")
    ax.text(2010, 0.045, "Autopsy ADNC (0-3) $\\eta^2$\n(0.002 - 0.009, n=1,166)", fontsize=8.3, color=PALETTE["crimson"], fontweight="bold")
    ax.set_ylim(-0.02, 0.62)
    ax.set_xlabel("Calendar Cutoff Year")
    ax.set_ylabel("Between-Cluster Variance Explained ($\\eta^2$)")
    ax.set_title("B. Clinical Progression vs. Autopsy ADNC ($\\eta^2$)")
    ax.legend(frameon=True, loc="center right", fontsize=7.5)

    # Panel C: Final 2026 Reference Severe-vs-Mild Risk Difference across 4 Autopsy Gold Standards
    ax = axes[2]
    np_cols = [
        ("np_adnc_int_high_diff", "ADNC\nInt/High"),
        ("np_braak_v_vi_diff", "Braak\nV-VI"),
        ("np_thal_4_5_diff", "Thal\n4-5"),
        ("np_lewy_any_diff", "Lewy\nBody"),
    ]
    x = np.arange(len(np_cols))
    w = 0.25
    cw_fin = fin[fin["panel_id"] == "NACC_P3_uds_crosswalk_8"].set_index("algo")
    for ai, (al, clr, lbl) in enumerate([("kmeans", PALETTE["navy"], "K-means"), ("gmm_diag", PALETTE["teal"], "GMM"), ("ward", PALETTE["coral"], "Ward")]):
        vals = [cw_fin.loc[al, c] for c, _ in np_cols]
        bars = ax.bar(x + (ai - 1) * w, vals, width=w, color=clr, label=lbl)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0.6 if v >= 0 else -1.6), f"{v:.1f}", ha="center", va="bottom", fontsize=7.2)
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in np_cols])
    ax.set_ylim(-4, 22)
    ax.set_ylabel("Severe minus Mild Cluster Risk Diff. (pp)")
    ax.set_title("C. Autopsy Pathology Contrast (Crosswalk, K=3)")
    ax.legend(frameon=True, loc="upper right", fontsize=7.5)

    fig.tight_layout()
    _save(fig, fig_dir, "fig2_prognosis_neuropathology_dissociation")


def make_fig3(res_dir: str, fig_dir: str) -> None:
    """Figure 3: ADNI 20-year EMCI Recruitment Wave & Unified FS7 vs Historical FS Pipeline."""
    df_a = pd.read_csv(os.path.join(res_dir, "trajectory_a_calendar_cutoff.csv"))
    df_a["yr"] = _yr_col(df_a)
    adni_k3 = df_a[(df_a["cohort"] == "ADNI") & (df_a["k"] == 3)]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.6))
    p_styles = [
        ("ADNI_P1_clinical_8", PALETTE["slate"], "-", "o", "Clinical 8-feature"),
        ("ADNI_P2_clin_fs7_13", PALETTE["teal"], "-", "D", "Clin + Unified FS7 (13 feat.)"),
        ("ADNI_P3_clin_fshist_13", PALETTE["coral"], "--", "s", "Clin + Historical FS 4.3-7.x"),
    ]

    # Panel A: K-means ARI across 3 ADNI panels
    ax = axes[0, 0]
    for pid, clr, ls, mk, lbl in p_styles:
        s = adni_k3[(adni_k3["panel_id"] == pid) & (adni_k3["algo"] == "kmeans")].sort_values("yr")
        ax.plot(s["yr"], s["ari_to_ref"], color=clr, ls=ls, marker=mk, ms=4, lw=2.0, label=lbl)
    ax.axvspan(2010, 2012, color=PALETTE["blue"], alpha=0.10, label="ADNIGO/2 EMCI wave")
    ax.set_ylim(0.30, 1.05)
    ax.set_title("A. ADNI K-means Stability Across 20 Years (K=3)")
    ax.set_xlabel("Calendar Cutoff Year")
    ax.set_ylabel("ARI vs. 2026 Reference")

    # Panel B: GMM ARI comparing Unified FS7 vs Historical FS
    ax = axes[0, 1]
    for pid, clr, ls, mk, lbl in p_styles:
        s = adni_k3[(adni_k3["panel_id"] == pid) & (adni_k3["algo"] == "gmm_diag")].sort_values("yr")
        ax.plot(s["yr"], s["ari_to_ref"], color=clr, ls=ls, marker=mk, ms=4, lw=2.0, label=lbl)
    ax.axvspan(2010, 2012, color=PALETTE["blue"], alpha=0.10)
    ax.set_ylim(0.25, 1.05)
    ax.set_title("B. ADNI Diagonal GMM: Unified FS7 vs. Historical FS (K=3)")
    ax.set_xlabel("Calendar Cutoff Year")
    ax.set_ylabel("ARI vs. 2026 Reference")

    # Panel C: ADNI Mean 2008-2025 Flip Rate by Algorithm x K (Clin+FS7)
    ax = axes[1, 0]
    pre_fs7 = df_a[(df_a["panel_id"] == "ADNI_P2_clin_fs7_13") & (df_a["cutoff_label"] != "2026_final")]
    ks = [2, 3, 4]
    x = np.arange(len(ks))
    w = 0.25
    for ai, (al, clr, lbl) in enumerate([("kmeans", PALETTE["navy"], "K-means"), ("gmm_diag", PALETTE["teal"], "Diagonal GMM"), ("ward", PALETTE["coral"], "Ward")]):
        vals = [pre_fs7[(pre_fs7["algo"] == al) & (pre_fs7["k"] == k)]["flip_rate_to_ref"].mean() * 100.0 for k in ks]
        bars = ax.bar(x + (ai - 1) * w, vals, width=w, color=clr, label=lbl)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["K = 2", "K = 3", "K = 4"])
    ax.set_ylim(0, 31)
    ax.set_title("C. ADNI Clin+FS7 Mean Flip Rate by Algorithm and K")
    ax.set_ylabel("Mean 2008-2025 Flip Rate (%)")
    ax.legend(frameon=True, loc="upper left")

    # Panel D: ADNI 24m MCI->Dementia Risk Difference (Severe - Mild) over 20 years (Clin+FS7, K=3)
    ax = axes[1, 1]
    for al, clr, ls, lbl in [("kmeans", PALETTE["navy"], "-", "K-means"), ("gmm_diag", PALETTE["teal"], "--", "Diagonal GMM"), ("ward", PALETTE["coral"], ":", "Ward")]:
        s = adni_k3[(adni_k3["panel_id"] == "ADNI_P2_clin_fs7_13") & (adni_k3["algo"] == al)].sort_values("yr")
        ax.plot(s["yr"], s["out_mci_conv_24m_diff"], color=clr, ls=ls, lw=2.0, marker="o", ms=3.5, label=lbl)
    ax.set_ylim(45, 98)
    ax.set_title("D. ADNI 24m MCI-to-Dementia Separation (Clin+FS7, K=3)")
    ax.set_xlabel("Calendar Cutoff Year")
    ax.set_ylabel("Severe minus Mild Conversion Risk Diff. (pp)")
    ax.legend(frameon=True, loc="upper right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, fig_dir, "fig3_adni_emci_wave_and_fs_pipeline")


def make_supp_figures(res_dir: str, fig_dir: str) -> None:
    """Supplementary Figures S1 (Trajectory B), S2 (54-strategy heatmap), S3 (Autopsy trajectories)."""
    df_a = pd.read_csv(os.path.join(res_dir, "trajectory_a_calendar_cutoff.csv"))
    df_a["yr"] = _yr_col(df_a)
    df_b = pd.read_csv(os.path.join(res_dir, "trajectory_b_chronological_doubling.csv"))

    # Fig S1: Trajectory B across all 6 panels (K=3)
    panels = [
        ("ADNI_P1_clinical_8", "ADNI Clinical (8)"),
        ("ADNI_P2_clin_fs7_13", "ADNI Clin+FS7 (13)"),
        ("ADNI_P3_clin_fshist_13", "ADNI Clin+FS Hist (13)"),
        ("NACC_P1_stable_7", "NACC Stable Core (7)"),
        ("NACC_P2_uds_switch_naive_9", "NACC Naive Switch (9)"),
        ("NACC_P3_uds_crosswalk_8", "NACC Crosswalk (8)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8))
    for ax, (pid, title) in zip(axes.flat, panels):
        sub = df_b[(df_b["panel_id"] == pid) & (df_b["k"] == 3)]
        for al, clr, lbl in [("kmeans", PALETTE["navy"], "K-means"), ("gmm_diag", PALETTE["teal"], "Diagonal GMM"), ("ward", PALETTE["coral"], "Ward")]:
            s = sub[sub["algo"] == al].sort_values("train_n")
            ax.plot(s["train_n"], s["ari_to_ref"], color=clr, lw=2.0, marker="o", ms=4, label=lbl)
            if "ari_to_ref_p025" in s.columns:
                ax.fill_between(s["train_n"], s["ari_to_ref_p025"], s["ari_to_ref_p975"], color=clr, alpha=0.14)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        xmin, xmax = sub["train_n"].min(), sub["train_n"].max()
        ticks = [t for t in [60, 150, 300, 600, 1200, 2500, 5000, 10000, 20000] if xmin * 0.8 <= t <= xmax * 1.2]
        if len(ticks) > 5:
            ticks = ticks[::2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,}" if t >= 1000 else str(t) for t in ticks], fontsize=8.0)
        ax.set_ylim(0.05, 1.05)
        ax.set_title(title)
        ax.set_xlabel("Chronological Train N (log)")
        ax.set_ylabel("ARI vs. 2026 Reference")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, fig_dir, "figS1_trajectory_b_doubling")

    # Fig S2: 54-strategy heatmap (mean ARI & Flip Rate across 2008-2025)
    pre = df_a[df_a["cutoff_label"] != "2026_final"]
    p_ord = [p for p, _ in panels]
    p_lbls = [t for _, t in panels]
    cols_ord = [(al, k) for al in ("kmeans", "gmm_diag", "ward") for k in (2, 3, 4)]
    col_lbls = [f"{'KM' if a == 'kmeans' else ('GMM' if a == 'gmm_diag' else 'Ward')}\nK={k}" for a, k in cols_ord]

    mat_ari = np.zeros((len(p_ord), len(cols_ord)))
    mat_flip = np.zeros((len(p_ord), len(cols_ord)))
    for i, pid in enumerate(p_ord):
        for j, (al, k) in enumerate(cols_ord):
            s = pre[(pre["panel_id"] == pid) & (pre["algo"] == al) & (pre["k"] == k)]
            mat_ari[i, j] = s["ari_to_ref"].mean()
            mat_flip[i, j] = s["flip_rate_to_ref"].mean() * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.6))
    im1 = ax1.imshow(mat_ari, cmap="YlGnBu", vmin=0.35, vmax=0.98, aspect="auto")
    ax1.set_xticks(range(len(col_lbls))); ax1.set_xticklabels(col_lbls, fontsize=8)
    ax1.set_yticks(range(len(p_lbls))); ax1.set_yticklabels(p_lbls, fontsize=8.5)
    ax1.set_title("A. Mean 2008-2025 ARI vs. 2026 Reference (54 Strategies)")
    ax1.grid(False)
    for i in range(mat_ari.shape[0]):
        for j in range(mat_ari.shape[1]):
            v = mat_ari[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.75 else "black")
    fig.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)

    im2 = ax2.imshow(mat_flip, cmap="YlOrRd", vmin=0.0, vmax=42.0, aspect="auto")
    ax2.set_xticks(range(len(col_lbls))); ax2.set_xticklabels(col_lbls, fontsize=8)
    ax2.set_yticks(range(len(p_lbls))); ax2.set_yticklabels([""] * len(p_lbls))
    ax2.set_title("B. Mean 2008-2025 Individual Flip Rate (%)")
    ax2.grid(False)
    for i in range(mat_flip.shape[0]):
        for j in range(mat_flip.shape[1]):
            v = mat_flip[i, j]
            ax2.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white" if v > 22.0 else "black")
    fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    fig.tight_layout()
    _save(fig, fig_dir, "figS2_54_strategy_stability_heatmap")

    # Fig S3: Autopsy neuropathology risk differences over 20 years (NACC Crosswalk vs Naive, K=3)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    np_specs = [
        ("np_adnc_int_high_diff", "A. Autopsy ADNC Int/High (NPADNC >= 2)"),
        ("np_braak_v_vi_diff", "B. Autopsy Braak Stage V-VI"),
        ("np_thal_4_5_diff", "C. Autopsy Thal Amyloid Phase 4-5"),
        ("np_lewy_any_diff", "D. Autopsy Lewy Body Co-Pathology"),
    ]
    nacc_k3 = df_a[(df_a["cohort"] == "NACC") & (df_a["k"] == 3)]
    for ax, (col, title) in zip(axes.flat, np_specs):
        ax.axhline(0.0, color="black", lw=0.9, ls=":")
        for pid, al, clr, ls, lbl in [
            ("NACC_P3_uds_crosswalk_8", "kmeans", PALETTE["navy"], "-", "Crosswalk K-means"),
            ("NACC_P3_uds_crosswalk_8", "gmm_diag", PALETTE["teal"], "-", "Crosswalk GMM"),
            ("NACC_P2_uds_switch_naive_9", "gmm_diag", PALETTE["crimson"], "--", "Naive Switch GMM"),
            ("NACC_P1_stable_7", "kmeans", PALETTE["slate"], ":", "Stable Core K-means"),
        ]:
            s = nacc_k3[(nacc_k3["panel_id"] == pid) & (nacc_k3["algo"] == al)].sort_values("yr")
            ax.plot(s["yr"], s[col], color=clr, ls=ls, lw=1.9, marker="o", ms=3.0, label=lbl)
        ax.axvline(2015, color=PALETTE["blue"], ls="-.", lw=0.9, alpha=0.55)
        ax.set_title(title)
        ax.set_xlabel("Calendar Cutoff Year")
        ax.set_ylabel("Severe - Mild Risk Diff. (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, fig_dir, "figS3_neuropathology_over_time")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir", default="outputs/subtyping_full")
    parser.add_argument("--fig-dir", default="outputs/figures")
    args = parser.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    _setup_style()
    make_fig1(args.res_dir, args.fig_dir)
    make_fig2(args.res_dir, args.fig_dir)
    make_fig3(args.res_dir, args.fig_dir)
    make_supp_figures(args.res_dir, args.fig_dir)
    print(f"Generated 6 subtyping figures in {args.fig_dir}")


if __name__ == "__main__":
    main()
