"""Read-only dual-cohort builder (ADNI Route A + NACC Freeze 75) for dual-cohort ADNI/NACC unsupervised subtyping & neuropathology validity."""
from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd

from subtyping_engine import apply_monsell_moca_to_mmse


ADNI_PHASES = ("ADNI1", "ADNIGO", "ADNI2", "ADNI3", "ADNI4")
FAQ_ITEMS_NACC = [
    "BILLS", "TAXES", "SHOPPING", "GAMES", "STOVE",
    "MEALPREP", "EVENTS", "PAYATTN", "REMDATES", "TRAVEL",
]

# Panel specifications: (feature_names, severity_signs: +1 higher=worse, -1 higher=better)
PANEL_SPECS: Dict[str, Tuple[List[str], List[float]]] = {
    "ADNI_P1_clinical_8": (
        ["cdrsb", "faq", "mmse", "ldel", "ravlt_imm", "traila", "trailb", "animals"],
        [1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
    ),
    "ADNI_P2_clin_fs7_13": (
        [
            "cdrsb", "faq", "mmse", "ldel", "ravlt_imm", "traila", "trailb", "animals",
            "fs7_hipp_icv", "fs7_ent_icv", "fs7_ent_thick", "fs7_vent_icv", "fs7_midtemp_thick",
        ],
        [1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0],
    ),
    "ADNI_P3_clin_fshist_13": (
        [
            "cdrsb", "faq", "mmse", "ldel", "ravlt_imm", "traila", "trailb", "animals",
            "fsh_hipp_icv", "fsh_ent_icv", "fsh_ent_thick", "fsh_vent_icv", "fsh_midtemp_thick",
        ],
        [1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0],
    ),
    "NACC_P1_stable_7": (
        ["cdrsb", "faq", "gds", "traila", "trailb", "animals", "veg"],
        [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
    ),
    "NACC_P2_uds_switch_naive_9": (
        ["cdrsb", "faq", "gds", "traila", "trailb", "animals", "veg", "mmse", "moca"],
        [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    ),
    "NACC_P3_uds_crosswalk_8": (
        ["cdrsb", "faq", "gds", "traila", "trailb", "animals", "veg", "mmse_crosswalk"],
        [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
    ),
}


@dataclass
class SubtypingCohortBundle:
    cohort: str
    df: pd.DataFrame  # subject-level baseline features + is_eval + external outcomes (clinical & neuropath)
    panel_ids: List[str]
    external_outcomes: List[Tuple[str, bool]]  # (col_name, is_binary)
    final_cutoff: pd.Timestamp


def _clean_range(s: pd.Series, lo: float, hi: float) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    return v.where((v >= lo) & (v <= hi), np.nan)


def _deterministic_stratified_holdout(
    subj_ids: pd.Series, strata: pd.Series, seed_tag: str, frac: float = 0.30
) -> np.ndarray:
    df = pd.DataFrame({"id": subj_ids.astype(str).values, "stratum": strata.astype(str).values})
    df["_hash"] = [hashlib.sha256(f"{seed_tag}|{sid}".encode("utf-8")).hexdigest() for sid in df["id"]]
    is_eval = np.zeros(len(df), dtype=bool)
    for _, grp in df.groupby("stratum", sort=True):
        order = grp.sort_values(["_hash", "id"]).index.values
        n_eval = int(np.floor(len(order) * frac + 0.5))
        if len(order) >= 2 and n_eval == 0:
            n_eval = 1
        is_eval[order[:n_eval]] = True
    return is_eval


def _load_fs_features(csv_path: str, base_bl: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Extract baseline (<=180d of bl_dt) ICV-normalized volumes & cortical thicknesses from a FreeSurfer table."""
    cols = [
        "RID", "EXAMDATE", "ST10CV", "ST29SV", "ST88SV", "ST24CV", "ST83CV",
        "ST24TA", "ST83TA", "ST37SV", "ST96SV", "ST30SV", "ST89SV", "ST40TA", "ST99TA",
    ]
    fs = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    fs["RID"] = pd.to_numeric(fs["RID"], errors="coerce")
    fs["visit_dt"] = pd.to_datetime(fs["EXAMDATE"], errors="coerce")
    fs = fs.dropna(subset=["RID", "visit_dt"]).copy()
    fs["RID"] = fs["RID"].astype(int)
    fs = fs.merge(base_bl[["RID", "bl_dt"]], on="RID", how="inner")
    fs["diff"] = (fs["visit_dt"] - fs["bl_dt"]).dt.days.abs()
    fs = fs[fs["diff"] <= 180].sort_values(["RID", "diff"])

    icv = _clean_range(fs["ST10CV"], 5e5, 3e6)
    hipp = _clean_range(fs["ST29SV"], 500, 15000) + _clean_range(fs["ST88SV"], 500, 15000)
    ent_v = _clean_range(fs["ST24CV"], 200, 10000) + _clean_range(fs["ST83CV"], 200, 10000)
    ent_t = 0.5 * (_clean_range(fs["ST24TA"], 0.5, 6.0) + _clean_range(fs["ST83TA"], 0.5, 6.0))
    vent = (
        _clean_range(fs["ST37SV"], 100, 200000) + _clean_range(fs["ST96SV"], 100, 200000)
        + _clean_range(fs["ST30SV"], 0, 50000) + _clean_range(fs["ST89SV"], 0, 50000)
    )
    mid_t = 0.5 * (_clean_range(fs["ST40TA"], 0.5, 6.0) + _clean_range(fs["ST99TA"], 0.5, 6.0))

    fs[f"{prefix}_hipp_icv"] = (hipp / icv) * 1000.0
    fs[f"{prefix}_ent_icv"] = (ent_v / icv) * 1000.0
    fs[f"{prefix}_ent_thick"] = ent_t
    fs[f"{prefix}_vent_icv"] = (vent / icv) * 100.0
    fs[f"{prefix}_midtemp_thick"] = mid_t
    out_cols = [f"{prefix}_hipp_icv", f"{prefix}_ent_icv", f"{prefix}_ent_thick", f"{prefix}_vent_icv", f"{prefix}_midtemp_thick"]
    # Keep first nearest scan that has non-null hippocampus
    fs["_has_h"] = fs[f"{prefix}_hipp_icv"].notna().astype(int)
    fs = fs.sort_values(["RID", "_has_h", "diff"], ascending=[True, False, True])
    return fs.groupby("RID", as_index=False).first()[["RID"] + out_cols]


def build_adni_subtyping_cohort(adni_raw_dir: str, adni_extract_dir: str) -> Tuple[SubtypingCohortBundle, Dict]:
    """Build enrolled ADNI impaired cohort (baseline MCI + Dementia) with clinical, FS7, and historical FS panels + 2y clinical outcomes."""
    v26 = os.path.join(adni_extract_dir, "v2026_10", "csv")
    adsl = pd.read_csv(os.path.join(v26, "ADSL.csv"), low_memory=False)
    adsl["RID"] = pd.to_numeric(adsl["SUBJID"], errors="coerce")
    enrl = adsl[(adsl["ENRLFL"] == "Y") & (adsl["ORIGPROT"].isin(ADNI_PHASES))].dropna(subset=["RID"]).drop_duplicates("RID").copy()
    enrl["RID"] = enrl["RID"].astype(int)

    dx_candidates = sorted(glob.glob(os.path.join(adni_raw_dir, "DXSUM*.csv")))
    if not dx_candidates:
        raise FileNotFoundError(f"No DXSUM*.csv found under {adni_raw_dir}")
    dx = pd.read_csv(dx_candidates[-1], low_memory=False)
    dx["RID"] = pd.to_numeric(dx["RID"], errors="coerce")
    dx["visit_dt"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx["dx"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")
    dx = dx[dx["RID"].isin(enrl["RID"]) & dx["visit_dt"].notna() & dx["dx"].isin([1, 2, 3])].copy()
    dx["RID"] = dx["RID"].astype(int)
    dx = dx.sort_values(["RID", "visit_dt"])

    bl_dx = dx.groupby("RID", as_index=False).first().rename(columns={"visit_dt": "bl_dt", "dx": "bl_dx"})
    base = enrl[["RID", "ORIGPROT", "AGE", "SEX", "EDUC"]].merge(bl_dx[["RID", "bl_dt", "bl_dx"]], on="RID", how="inner")
    # Restrict to cognitively impaired spectrum (MCI=2, Dementia=3)
    base = base[base["bl_dx"].isin([2, 3])].copy().reset_index(drop=True)

    # Extract clinical baseline features from ADQS & NEUROBAT
    adqs = pd.read_csv(os.path.join(v26, "ADQS.csv"), usecols=["SUBJID", "PARAMCD", "AVAL", "ADT", "ABLFL"], low_memory=False)
    adqs["RID"] = pd.to_numeric(adqs["SUBJID"], errors="coerce")
    adqs = adqs[adqs["RID"].isin(base["RID"]) & adqs["PARAMCD"].isin(["CDRSB", "FAQTOTAL", "MMSCORE", "LDELTOTL", "RAVLTIMM", "TRABSCOR"])].copy()
    adqs["RID"] = adqs["RID"].astype(int)
    adqs["visit_dt"] = pd.to_datetime(adqs["ADT"], errors="coerce")
    adqs["AVAL"] = pd.to_numeric(adqs["AVAL"], errors="coerce")
    adqs = adqs.dropna(subset=["AVAL", "visit_dt"])

    base = base.set_index("RID")
    for pcd, cname, lo, hi in [
        ("CDRSB", "cdrsb", 0.0, 18.0),
        ("FAQTOTAL", "faq", 0.0, 30.0),
        ("MMSCORE", "mmse", 0.0, 30.0),
        ("LDELTOTL", "ldel", 0.0, 25.0),
        ("RAVLTIMM", "ravlt_imm", 0.0, 75.0),
        ("TRABSCOR", "trailb", 0.0, 300.0),
    ]:
        sub = adqs[adqs["PARAMCD"] == pcd].copy()
        sub["val"] = _clean_range(sub["AVAL"], lo, hi)
        sub = sub.dropna(subset=["val"]).merge(base[["bl_dt"]].reset_index(), on="RID", how="inner")
        sub["diff"] = (sub["visit_dt"] - sub["bl_dt"]).dt.days.abs()
        sub["is_abl"] = (sub["ABLFL"] == "Y").astype(int)
        sub = sub[(sub["is_abl"] == 1) | (sub["diff"] <= 180)].sort_values(["RID", "is_abl", "diff"], ascending=[True, False, True])
        base[cname] = sub.groupby("RID")["val"].first()

    nb = pd.read_csv(os.path.join(v26, "NEUROBAT.csv"), usecols=["RID", "VISDATE", "TRAASCOR", "TRABSCOR", "CATANIMSC"], low_memory=False)
    nb["RID"] = pd.to_numeric(nb["RID"], errors="coerce")
    nb = nb[nb["RID"].isin(base.index)].copy()
    nb["RID"] = nb["RID"].astype(int)
    nb["visit_dt"] = pd.to_datetime(nb["VISDATE"], errors="coerce")
    nb = nb.dropna(subset=["visit_dt"]).merge(base[["bl_dt"]].reset_index(), on="RID", how="inner")
    nb["diff"] = (nb["visit_dt"] - nb["bl_dt"]).dt.days.abs()
    nb = nb[nb["diff"] <= 180].sort_values(["RID", "diff"])
    for raw_c, out_c, lo, hi in [("TRAASCOR", "traila", 0.0, 150.0), ("CATANIMSC", "animals", 0.0, 77.0), ("TRABSCOR", "trailb_nb", 0.0, 300.0)]:
        tmp = nb.assign(_v=_clean_range(nb[raw_c], lo, hi)).dropna(subset=["_v"]).groupby("RID")["_v"].first()
        if out_c == "trailb_nb":
            base["trailb"] = base["trailb"].fillna(tmp)
        else:
            base[out_c] = tmp

    base = base.reset_index()

    # Unified FS7 features across all 5 phases
    fs7_df = _load_fs_features(os.path.join(v26, "UCSFFSX7.csv"), base[["RID", "bl_dt"]], "fs7")
    base = base.merge(fs7_df, on="RID", how="left")

    # Historical phase-native FreeSurfer pipeline (FS4.3 ADNI1, FS5.1 ADNIGO/2, FS6.0 ADNI3, FS7.x ADNI4)
    fs43 = _load_fs_features(os.path.join(v26, "UCSFFSX.csv"), base[base["ORIGPROT"] == "ADNI1"][["RID", "bl_dt"]], "fsh")
    fs51 = _load_fs_features(os.path.join(v26, "UCSFFSX51.csv"), base[base["ORIGPROT"].isin(["ADNIGO", "ADNI2"])][["RID", "bl_dt"]], "fsh")
    fs60 = _load_fs_features(os.path.join(v26, "UCSFFSX6.csv"), base[base["ORIGPROT"] == "ADNI3"][["RID", "bl_dt"]], "fsh")
    fs7_a4 = _load_fs_features(os.path.join(v26, "UCSFFSX7.csv"), base[base["ORIGPROT"] == "ADNI4"][["RID", "bl_dt"]], "fsh")
    fsh_all = pd.concat([fs43, fs51, fs60, fs7_a4], ignore_index=True).drop_duplicates("RID")
    base = base.merge(fsh_all, on="RID", how="left")

    # External 24m clinical outcomes (strict window closure <= 2026-11-20; never used in clustering)
    final_cut = pd.Timestamp("2026-11-20")
    win_closed = (base["bl_dt"] + pd.to_timedelta(27.0 * 30.4375, unit="D")) <= final_cut

    # 1. 24m MCI -> Dementia conversion (only among baseline MCI with closed 27m window)
    dx_m = dx.merge(base[["RID", "bl_dt", "bl_dx"]], on="RID", how="inner")
    dx_m["fu_m"] = (dx_m["visit_dt"] - dx_m["bl_dt"]).dt.days / 30.4375
    dx_pos = dx_m[dx_m["fu_m"] > 0]
    ev_t = dx_pos[dx_pos["dx"] == 3].groupby("RID")["fu_m"].min()
    max_fu = dx_pos.groupby("RID")["fu_m"].max()
    base = base.set_index("RID")
    base["_ev_t"] = ev_t
    base["_max_fu"] = max_fu
    mci_closed = (base["bl_dx"] == 2) & ((base["bl_dt"] + pd.to_timedelta(27.0 * 30.4375, unit="D")) <= final_cut)
    base["out_mci_conv_24m"] = np.where(
        mci_closed & (base["_ev_t"] <= 27.0), 1.0,
        np.where(mci_closed & (base["_max_fu"] >= 21.0), 0.0, np.nan)
    )

    # 2. 24m CDR-SB & FAQ (closest visit in 18-30m window, strict window closure)
    for fname, dcol, scol, param_adqs, out_col, lo, hi in [
        ("CDR.csv", "VISDATE", "CDRSB", "CDRSB", "out_cdrsb_24m", 0.0, 18.0),
        ("FAQ.csv", "VISDATE", "FAQTOTAL", "FAQTOTAL", "out_faq_24m", 0.0, 30.0),
    ]:
        r_t = pd.read_csv(os.path.join(v26, fname), usecols=["RID", dcol, scol], low_memory=False)
        r_t["RID"] = pd.to_numeric(r_t["RID"], errors="coerce")
        r_t["visit_dt"] = pd.to_datetime(r_t[dcol], errors="coerce")
        r_t["val"] = _clean_range(r_t[scol], lo, hi)
        a_t = adqs[adqs["PARAMCD"] == param_adqs][["RID", "visit_dt", "AVAL"]].rename(columns={"AVAL": "val"})
        comb = pd.concat([r_t[["RID", "visit_dt", "val"]], a_t], ignore_index=True).dropna()
        comb["RID"] = comb["RID"].astype(int)
        comb = comb.merge(base[["bl_dt"]].reset_index(), on="RID", how="inner")
        comb["fu_m"] = (comb["visit_dt"] - comb["bl_dt"]).dt.days / 30.4375
        w = comb[(comb["fu_m"] >= 18.0) & (comb["fu_m"] <= 30.0)].copy()
        w["dist"] = (w["fu_m"] - 24.0).abs()
        w24 = w.sort_values(["RID", "dist", "fu_m"]).groupby("RID")["val"].first()
        base[out_col] = w24
        base.loc[~((base["bl_dt"] + pd.to_timedelta(30.0 * 30.4375, unit="D")) <= final_cut), out_col] = np.nan

    base["out_dcdrsb_24m"] = base["out_cdrsb_24m"] - base["cdrsb"]
    base = base.reset_index()
    base["subj_id"] = base["RID"].map(lambda r: f"ADNI_{int(r):05d}")
    base["phase_ver"] = base["ORIGPROT"].astype(str)
    base["bl_year"] = base["bl_dt"].dt.year.astype(int)
    base["stratum"] = base["phase_ver"] + "_" + base["bl_dx"].astype(int).astype(str)
    base["is_eval"] = _deterministic_stratified_holdout(
        base["subj_id"], base["stratum"], seed_tag="LC02-adni-subtyping-eval-20261123", frac=0.30
    )

    ext_outs = [
        ("out_mci_conv_24m", True),
        ("out_cdrsb_24m", False),
        ("out_dcdrsb_24m", False),
        ("out_faq_24m", False),
    ]
    bundle = SubtypingCohortBundle(
        cohort="ADNI",
        df=base.sort_values("subj_id").reset_index(drop=True),
        panel_ids=["ADNI_P1_clinical_8", "ADNI_P2_clin_fs7_13", "ADNI_P3_clin_fshist_13"],
        external_outcomes=ext_outs,
        final_cutoff=final_cut,
    )
    meta = {
        "cohort": "ADNI",
        "n_impaired": int(len(base)),
        "n_eval": int(base["is_eval"].sum()),
        "by_phase": base["phase_ver"].value_counts().to_dict(),
        "outcome_n_eval": {c: int(base.loc[base["is_eval"], c].notna().sum()) for c, _ in ext_outs},
    }
    return bundle, meta


def build_nacc_subtyping_cohort(nacc_parquet: str) -> Tuple[SubtypingCohortBundle, Dict]:
    """Build NACC Freeze 75 impaired cohort (MCI + Dementia) with UDS switch/crosswalk panels, 2y clinical outcomes, and autopsy neuropathology."""
    cols = [
        "NACCID", "NACCVNUM", "VISITDATE", "FORMVER", "NACCAGE", "NACCSEX", "EDUC",
        "NACCUDSD", "CDRSUM", "NACCGDS", "TRAILA", "TRAILB", "ANIMALS", "VEG",
        "NACCMMSE", "NACCMOCA", "NPADNC", "NACCBRAA", "NPTHAL", "NACCLEWY",
    ] + FAQ_ITEMS_NACC
    df = pd.read_parquet(nacc_parquet, columns=cols)
    df["visit_dt"] = pd.to_datetime(df["VISITDATE"], errors="coerce")
    df = df.dropna(subset=["NACCID", "visit_dt"]).sort_values(["NACCID", "NACCVNUM"])

    df["cdrsb"] = _clean_range(df["CDRSUM"], 0.0, 18.0)
    faq_mat = np.column_stack([_clean_range(df[c], 0.0, 3.0).to_numpy(dtype=np.float64) for c in FAQ_ITEMS_NACC])
    faq_cnt = np.sum(np.isfinite(faq_mat), axis=1)
    with np.errstate(invalid="ignore"):
        faq_sum = np.nansum(faq_mat, axis=1)
        df["faq"] = np.where(faq_cnt >= 8, (faq_sum / np.maximum(faq_cnt, 1)) * 10.0, np.nan)

    # Baseline visit
    bl = df.groupby("NACCID", as_index=False).first().copy()
    bl["bl_dx"] = pd.to_numeric(bl["NACCUDSD"], errors="coerce").fillna(-1).astype(int)
    # Restrict to impaired spectrum (MCI=3, Dementia=4)
    bl = bl[bl["bl_dx"].isin([3, 4])].copy().reset_index(drop=True)
    imp_ids = set(bl["NACCID"])

    bl["subj_id"] = bl["NACCID"].astype(str)
    bl["bl_dt"] = bl["visit_dt"]
    bl["bl_year"] = bl["bl_dt"].dt.year.astype(int)
    bl["phase_ver"] = "UDS" + bl["FORMVER"].fillna(0).astype(int).astype(str)
    bl["gds"] = _clean_range(bl["NACCGDS"], 0.0, 15.0)
    bl["traila"] = _clean_range(bl["TRAILA"], 0.0, 150.0)
    bl["trailb"] = _clean_range(bl["TRAILB"], 0.0, 300.0)
    bl["animals"] = _clean_range(bl["ANIMALS"], 0.0, 77.0)
    bl["veg"] = _clean_range(bl["VEG"], 0.0, 77.0)
    bl["mmse"] = _clean_range(bl["NACCMMSE"], 0.0, 30.0)
    bl["moca"] = _clean_range(bl["NACCMOCA"], 0.0, 30.0)

    # Monsell et al. (2016) Table 3 static crosswalk: NACCMOCA is already education-adjusted!
    moca_eq = apply_monsell_moca_to_mmse(bl["moca"].to_numpy(dtype=np.float64), already_adj=True)
    bl["mmse_crosswalk"] = np.where(np.isfinite(bl["mmse"]), bl["mmse"], moca_eq)

    # Neuropathology gold standard (static autopsy record per NACCID, never in clustering)
    np_agg = df[df["NACCID"].isin(imp_ids)].groupby("NACCID").agg({
        "NPADNC": "first", "NACCBRAA": "first", "NPTHAL": "first", "NACCLEWY": "first"
    })
    bl = bl.set_index("NACCID")
    adnc = pd.to_numeric(np_agg["NPADNC"], errors="coerce")
    braak = pd.to_numeric(np_agg["NACCBRAA"], errors="coerce")
    thal = pd.to_numeric(np_agg["NPTHAL"], errors="coerce")
    lewy = pd.to_numeric(np_agg["NACCLEWY"], errors="coerce")

    bl["np_adnc_ord"] = adnc.where(adnc.isin([0, 1, 2, 3]), np.nan)
    bl["np_adnc_int_high"] = np.where(bl["np_adnc_ord"].notna(), (bl["np_adnc_ord"] >= 2).astype(float), np.nan)
    bl["np_braak_v_vi"] = np.where(braak.isin([0, 1, 2, 3, 4, 5, 6]), (braak >= 5).astype(float), np.nan)
    bl["np_thal_4_5"] = np.where(thal.isin([0, 1, 2, 3, 4, 5]), (thal >= 4).astype(float), np.nan)
    bl["np_lewy_any"] = np.where(lewy.isin([0, 1, 2, 3, 4]), (lewy >= 1).astype(float), np.nan)

    # 24m longitudinal clinical outcomes (strict window closure <= 2026-08-20)
    final_cut = pd.Timestamp("2026-08-20")
    sub_l = df[df["NACCID"].isin(imp_ids)][["NACCID", "visit_dt", "NACCUDSD", "cdrsb", "faq"]].copy()
    sub_l = sub_l.merge(bl[["bl_dt", "bl_dx"]].reset_index(), on="NACCID", how="inner")
    sub_l["fu_m"] = (sub_l["visit_dt"] - sub_l["bl_dt"]).dt.days / 30.4375

    pos_l = sub_l[sub_l["fu_m"] > 0]
    ev_t = pos_l[pos_l["NACCUDSD"] == 4].groupby("NACCID")["fu_m"].min()
    max_fu = pos_l.groupby("NACCID")["fu_m"].max()
    bl["_ev_t"] = ev_t
    bl["_max_fu"] = max_fu
    mci_closed = (bl["bl_dx"] == 3) & ((bl["bl_dt"] + pd.to_timedelta(27.0 * 30.4375, unit="D")) <= final_cut)
    bl["out_mci_conv_24m"] = np.where(
        mci_closed & (bl["_ev_t"] <= 27.0), 1.0,
        np.where(mci_closed & (bl["_max_fu"] >= 21.0), 0.0, np.nan)
    )

    w = sub_l[(sub_l["fu_m"] >= 18.0) & (sub_l["fu_m"] <= 30.0)].copy()
    w["dist"] = (w["fu_m"] - 24.0).abs()
    w = w.sort_values(["NACCID", "dist", "fu_m"])
    bl["out_cdrsb_24m"] = w.dropna(subset=["cdrsb"]).groupby("NACCID")["cdrsb"].first()
    bl["out_faq_24m"] = w.dropna(subset=["faq"]).groupby("NACCID")["faq"].first()
    not_closed_30 = ~((bl["bl_dt"] + pd.to_timedelta(30.0 * 30.4375, unit="D")) <= final_cut)
    bl.loc[not_closed_30, ["out_cdrsb_24m", "out_faq_24m"]] = np.nan
    bl["out_dcdrsb_24m"] = bl["out_cdrsb_24m"] - bl["cdrsb"]

    bl = bl.reset_index()
    # Stratify holdout by FORMVER x bl_dx x has_autopsy so autopsy gold standard is balanced in 30% eval pool!
    bl["_has_np"] = bl["np_adnc_ord"].notna().astype(int).astype(str)
    bl["stratum"] = bl["phase_ver"] + "_" + bl["bl_dx"].astype(str) + "_np" + bl["_has_np"]
    bl["is_eval"] = _deterministic_stratified_holdout(
        bl["subj_id"], bl["stratum"], seed_tag="LC02-nacc-subtyping-eval-20261123", frac=0.30
    )

    ext_outs = [
        ("out_mci_conv_24m", True),
        ("out_cdrsb_24m", False),
        ("out_dcdrsb_24m", False),
        ("out_faq_24m", False),
        ("np_adnc_ord", False),
        ("np_adnc_int_high", True),
        ("np_braak_v_vi", True),
        ("np_thal_4_5", True),
        ("np_lewy_any", True),
    ]
    bundle = SubtypingCohortBundle(
        cohort="NACC",
        df=bl.sort_values("subj_id").reset_index(drop=True),
        panel_ids=["NACC_P1_stable_7", "NACC_P2_uds_switch_naive_9", "NACC_P3_uds_crosswalk_8"],
        external_outcomes=ext_outs,
        final_cutoff=final_cut,
    )
    meta = {
        "cohort": "NACC",
        "n_impaired": int(len(bl)),
        "n_eval": int(bl["is_eval"].sum()),
        "by_phase": bl["phase_ver"].value_counts().to_dict(),
        "outcome_n_eval": {c: int(bl.loc[bl["is_eval"], c].notna().sum()) for c, _ in ext_outs},
    }
    return bundle, meta
