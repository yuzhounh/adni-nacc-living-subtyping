# ADNI + NACC Twenty-Year Unsupervised Subtyping Stability and Autopsy Neuropathology Validity

Standalone, pure-Python analytical pipeline for **Twenty-year cross-protocol stability, prognostic dissociation and autopsy neuropathology validity of unsupervised Alzheimer's disease subtypes in ADNI and NACC**.

## Overview

This repository evaluates 54 unsupervised subtyping strategies (6 feature panels x 3 clustering algorithms x K in {2, 3, 4}) across cognitively impaired baseline ADNI (`N = 1,750`) and NACC Freeze 75 (`N = 30,696`) cohorts over 2005--2026:

1. **NACC UDS 2.0 to 3.0 hard battery switch vs. Monsell (2016) crosswalk**: Contrasts a cross-UDS stable 7-feature core, naive unharmonized MMSE + MoCA switching, and static Monsell et al. (2016) Table 3 equipercentile crosswalk harmonization.
2. **ADNI unified FreeSurfer 7.x vs. historical phase-native pipelines**: Compares clinical-only, clinical plus unified `UCSFFSX7`, and clinical plus phase-native `FS 4.3 -> 5.1 -> 6.0 -> 7.x` panels.
3. **Independent clinical progression and autopsy neuropathology gold standards**: Evaluates clustering stability (ARI, NMI, Hungarian-aligned individual flip rates, step ARI, silhouette) jointly against 24-month clinical progression and autopsy `NPADNC`, `NACCBRAA`, `NPTHAL`, and `NACCLEWY` records in an autopsy-balanced 30% evaluation reserve.

## Repository Structure

- `subtyping_engine.py`: Leakage-free preprocessing, K-means, diagonal GMM, exact weighted agglomerative Ward (400-microcluster acceleration above `n = 3,500`), Hungarian training-centroid alignment, external validity metrics, and Monsell (2016) Table 3 lookup.
- `build_subtyping_cohorts.py`: Read-only ADNI and NACC impaired-cohort builder.
- `run_subtyping_evolution.py`: Parallel 54-strategy runner across Trajectories A and B with zero-drift anchor audits.
- `make_figures.py` / `make_tables.py`: Regenerate all manuscript figures and LaTeX tables from aggregate CSV outputs.
- `tests/test_subtyping_engine.py`: Four pure-synthetic unit tests (zero private data required).

## Quickstart

```bash
pip install -r requirements.txt

# Run pure-synthetic unit tests (no restricted data required)
python tests/test_subtyping_engine.py

# Run pilot or full 54-strategy pipeline against local ADNI/NACC data
python run_subtyping_evolution.py --pilot --workers 12 --out-dir outputs
python run_subtyping_evolution.py --n-reps 50 --workers 16 --out-dir outputs
```
