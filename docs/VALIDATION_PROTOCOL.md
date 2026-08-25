# G1–G5 executable validation protocol

## G1 — genome-wide geometry signal

PASS requires external incremental geometry `R² >= 0.02`, positive lower 95% CI, and at least three of five landmark classes with `ΔR² >= 0.01` plus multiplicity-controlled evidence.

## G2 — low-dimensional morphology prediction

For `k` preregistered morphology parameters, at least `ceil(k/2)` distinct parameters must show external `ΔR² >= 0.01` with adjusted `p < 0.05`. The multivariate vector must also have positive predictive ability.

## G3 — useful bottleneck

```text
nRMSE(G -> Θ -> L) <= nRMSE(covariates) - 0.01
```

## G4 — privileged representation versus PCA

```text
nRMSE(G -> MGC -> L) <= nRMSE(G -> PCA_k -> L) - 0.01
```

with identical `k`.

## G5 — genomic-morphological compression

```text
eta = ΔR²_MGC / ΔR²_direct
```

PASS requires all of:

- `eta >= 0.90`;
- `ΔR²_direct >= 0.05`;
- lower 95% CI of `eta >= 0.75`;
- `k <= 3`;
- valid external execution.

If `ΔR²_direct <= 0`, eta is undefined and the engine must not report a large ratio caused by two noise terms.

## Strong support

Strong support requires G1–G5 plus external transfer execution, QC and no leakage. It remains support for a candidate research hypothesis, not proof of a universal biological law.
