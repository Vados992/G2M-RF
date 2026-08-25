# G2M-RF Testing and Validation Standard

This repository distinguishes **software verification** from **empirical validation of the biological hypothesis**. A green CI run verifies the implementation and its numerical contracts; it does not imply that G1-G5 have passed on real human data.

## CI layers

### 1. Unit and mathematical invariants

The suite checks frozen constants and identities, geometry envelopes, coordinate normalization, planning equations, kernel PSD behavior, penalized-FWL equivalence, multiplicity correction and statistical metrics.

### 2. Numerical validation

Numerical tests cover inverse sample-size calculations, M4 parameter recovery, projection/residualization identities, exact fixed-effect handling and Exact-vs-full-rank-Nyström agreement on deterministic synthetic problems.

### 3. Integration

Both Exact and Nyström pipelines are executed end-to-end. The CLI is exercised through planning, synthetic data generation, confirmatory execution and strict JSON report creation.

### 4. Reproducibility

Synthetic generation, family-aware splitting, bootstrap replicates, Nyström landmark selection and complete report metrics are checked for deterministic behavior under fixed seeds. Reports include runtime provenance.

### 5. Failure and adversarial behavior

Malformed dimensions, non-finite geometry/covariates, invalid genotype dosages, monomorphic genotype matrices, invalid planning parameters, invalid solver settings, non-square kernels, singular geometry candidates and undersized confirmatory splits must fail early with explicit exceptions.

### 6. G1-G5 decision scenarios

Every gate has boundary and failure scenarios. The test suite verifies the exact ordering of failure labels, QC/no-leakage invalidation, compression thresholds, k constraints and an explicit transferability failure path.

### 7. Scaling guards

The Nyström backend is instrumented to ensure that a large-N code path does not silently allocate a full N x N kernel. Auto-selection must route large cohorts away from the Exact backend.

### 8. Packaging and deployment

CI builds a wheel, checks installed dependencies, runs the installed CLI, builds the Docker image and executes the CLI from the non-root container runtime.

## Compatibility matrix

The GitHub Actions matrix covers:

- Ubuntu / Python 3.10
- Ubuntu / Python 3.11 (quality and coverage job)
- Ubuntu / Python 3.12
- Windows / Python 3.11
- macOS / Python 3.11

## Coverage gate

The quality job enforces branch coverage through `pytest-cov`. The repository threshold is defined in `pyproject.toml`; falling below it fails CI.

## What CI does not prove

CI cannot establish biological truth, population transportability or PASS for G1-G5. Those require frozen real genotype + morphology datasets, pre-registered analysis, independent external cohorts, acquisition reliability measurements and the confirmatory protocol described in the research document.

## Local verification

```bash
python -m pip install -e ".[dev]"
pip check
ruff check src tests scripts
pytest --timeout=120 --cov=g2mrf --cov-branch --cov-report=term-missing
```

For deployment verification:

```bash
pip wheel . --no-deps -w dist
docker build -t g2mrf .
docker run --rm g2mrf plan --h2 0.35 --me 75000 --n 20000 --target 0.05
```
