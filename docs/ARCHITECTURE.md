# Architecture

## Scientific data flow

```text
Whole-genome variation G
        |
        v
TRAIN-only genomic representation Φ(G, P)
        |
        +------------------------------+
        |                              |
        v                              v
Direct branch G -> L              Bottleneck branch G -> Θ -> L
        |                              |
        |                              +--> compare against matched PCA_k
        +--------------+---------------+
                       v
                 External metrics
                       |
                       v
                   G1 ... G5
```

Morphology parameters `Θ` are fitted from geometry **without genotype access**. This prevents double fitting of the representation to genomic outcomes.

## Three prediction branches

### A. Direct geometry

A fixed-effect genomic regressor predicts all landmark radii directly. Incremental external `R²` over allowed covariates is used for G1 and as the denominator of G5.

### B. MGC bottleneck

The current executable confirmatory implementation uses M4 parameters `(a, b)` as a two-dimensional morphology bottleneck. Genome predicts `(a, b)`, after which the frozen M4 map reconstructs landmark radii.

The geometry module also implements all frozen families M0–M5, so a preregistered release can select another family without changing prediction code.

### C. Matched-dimensional PCA

PCA is fitted on TRAIN geometry only with `k = dim(Θ)`. Genomic prediction is trained on the TRAIN PCA scores and reconstructed into landmark space. G4 compares the bottleneck to this neutral statistical compression at equal dimension.

## Corrected covariate handling

For explicit features the penalized model is

```text
min_{gamma,beta} ||y - C gamma - X beta||² + lambda ||beta||²
```

with

```text
M_C = I - C(C^T C)^+ C^T
beta_hat = (X^T M_C X + lambda I)^-1 X^T M_C y
```

Equivalently both sides are residualized:

```text
y_r = M_C y
X_r = M_C X
```

The scalable Nyström implementation computes these projections without materializing `M_C` as an `N x N` matrix.

The exact kernel implementation uses an unpenalized fixed-effect block system:

```text
[K + lambda I, C] [alpha] = [y]
[C^T           , 0] [gamma]   [0]
```

and predicts `K_* alpha + C_* gamma`.

## Exact versus scalable solver

`ExactLinearKRR` forms the complete Gram matrix and is the reference implementation for tests and small studies.

`NystromLinearKRR` samples deterministic landmark participants, computes `N x m` kernel features and performs corrected FWL in the feature space. It avoids allocating the full kernel and is suitable as the local scalable backend.

For national-scale WGS studies, replace only the matrix backend with PLINK2/Hail/Spark/SLURM or another distributed implementation. Gate logic and frozen statistics remain unchanged.
