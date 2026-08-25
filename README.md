# G2M-RF v2.0

Executable research implementation of **G2M-RF — Genotype → Morphology Research Framework**.

**System architect:** Vadym Tsinderhoz  
**Status:** research software / pre-data confirmatory framework  
**Theory manifest:** `docs/THEORY_DOCUMENT.md` (canonical PDF filename + SHA-256)

This repository turns the mathematical protocol into reproducible code. It does **not** claim that the biological hypothesis is already validated. The code is designed to make the hypothesis testable and falsifiable on independent genotype + morphology cohorts.

## What is implemented

- frozen geometry families M0–M5;
- participant-level M4 fitting without genome access;
- train-only genotype standardization and imputation;
- linear genomic kernel and PSD checks;
- corrected covariate handling for penalized regression/KRR;
- exact fixed-effect KRR for small/medium studies;
- Nyström approximation for larger cohorts without an `N x N` kernel allocation;
- direct `G → L`, MGC `G → Θ → L`, and matched-dimensional `G → PCA_k → L` branches;
- G1–G5 decision engine with the v2.0 thresholds;
- paired participant bootstrap for incremental `R²` and compression ratio `η`;
- Holm multiplicity correction;
- deterministic family-aware split;
- Daetwyler-style **planning envelope** for N = 20k / 50k / 100k / 200k / 500k;
- synthetic end-to-end dataset and reproducible demo;
- unit/property tests, Docker image and GitHub Actions CI.

## Important scientific boundary

`g2mrf plan` is a **power/planning tool only**. Its predicted `R²` must never be substituted for an observed external `ΔR²` when evaluating G1–G5.

A PASS result requires real independent data. Synthetic demo results only verify software behavior.

## Installation

Run the project as a Python package. **Do not execute files inside `src/g2mrf/` directly** (for example, `python src/g2mrf/cli.py`), because the package uses relative imports.

```bash
git clone https://github.com/Vados992/G2M-RF.git
cd G2M-RF
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
g2mrf --help
```

If you prefer module execution after installation, use:

```bash
python -m g2mrf.cli --help
```

## Run the complete synthetic pipeline

```bash
g2mrf demo --config configs/default.yaml --out results/demo_report.json
```

The command performs a deterministic family-aware split into TRAIN / INTERNAL / EXTERNAL, estimates morphology parameters independently of genotype, fits all competing prediction branches, computes bootstrap evidence and returns the G1–G5 decision chain.

## Planning table

```bash
g2mrf plan --h2 0.35 --me 75000 --target 0.05
```

Example study sizes are calculated with

```text
R²_pred ≈ h² / (1 + Me / (N_train h²))
```

where `N_train` is **TRAIN size**, not the total internal cohort size.

## Run on an analysis-ready dataset

Create an `.npz` bundle containing:

- `sample_ids`: shape `(N,)`
- `family_ids`: shape `(N,)`
- `G`: genotype dosage matrix `(N, M)` with values 0/1/2 and optional NaN
- `C`: allowed covariates `(N, Q)`; do not include an intercept (the code adds it)
- `radii`: height-normalized radial landmarks `(N, L)`
- `angles`: fixed angles `(L,)` in radians
- `landmark_classes`: five or more preregistered class labels `(L,)`

Then run:

```bash
g2mrf run --data cohort.npz --config configs/default.yaml --out results/report.json
```

The input format is deliberately analysis-ready. Raw WGS calling, alignment, pangenome mapping, scanner reconstruction and ethics/consent systems belong upstream and must be version-locked before this package is invoked.

## Scaling

Exact KRR is useful as a mathematical reference implementation, but it allocates an `N x N` kernel. At 250,000 participants a float64 kernel alone would require roughly 500 GB before solver overhead.

For this reason `solver: auto` switches to Nyström features above `exact_threshold`. Production deployments at 100k–500k scale should use distributed genotype storage and block/streamed matrix multiplication; the public API intentionally separates the statistical model from storage so an HPC backend can replace the local NumPy backend without changing gate definitions.

## Repository layout

```text
src/g2mrf/
  geometry/       M0–M5, envelope checks, fitting, coordinate transforms
  genomics/       train-only standardization, kernels, exact/scalable KRR
  statistics/     metrics, bootstrap, multiplicity
  data/           analysis-ready IO and synthetic generator
  config.py       frozen thresholds/model settings
  planning.py     sample-size planning envelope
  gates.py        G1–G5 decision engine
  pipeline.py     end-to-end confirmatory execution
  cli.py          command-line interface

tests/            mathematical, statistical and pipeline tests
configs/          frozen example configuration
docs/             theory manifest and implementation documentation
scripts/          optional data conversion/publishing helpers
.github/workflows CI
```

## Reproducibility rules

1. Build variant dictionaries, allele frequencies, scalers and population features on TRAIN only.
2. Keep entire kinship/family components in one split.
3. Estimate morphology parameters without genotype access.
4. Residualize both phenotype and genomic design when using penalized FWL; never residualize only `y`.
5. Do not convert cross-validated `lambda` into an estimate of heritability.
6. Do not attenuate an observed-scale published `h²` by reliability a second time.
7. Lock model definitions and thresholds before INTERNAL TEST.
8. Open INTERNAL and EXTERNAL only once for a confirmatory release.
9. Report negative results as valid scientific outcomes.

## License

Copyright © 2026 Vadym Tsinderhoz. See `LICENSE`.
