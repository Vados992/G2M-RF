# Verification record

Verification executed in the build environment on 2026-08-25.

## Automated tests

```text
20 passed
```

Coverage areas include:

- golden-spiral constant and physical envelope;
- scale invariance;
- M0 analytic optimum;
- M1 physical range;
- logarithmic-spiral phase identity;
- ellipse positivity and parabola singularity rejection;
- genomic-kernel PSD and symmetry;
- corrected covariate-residualized kernel;
- penalized-FWL counterexample correction (`beta = 4/3`);
- TRAIN-only genotype preprocessing;
- deterministic family-aware split;
- deterministic participant bootstrap;
- Holm correction;
- exact and Nyström genomic regressors;
- G1–G5 decision chain;
- end-to-end train/internal/external pipeline.

## End-to-end synthetic validation

A 720-participant / 500-variant / 25-landmark synthetic run completed successfully.

Observed software decision:

```text
G1 PASS
G2 PASS
G3 PASS
G4 FAIL
G5 PASS
Strong support: FALSE
Decision: MGC NOT PRIVILEGED VS PCA
```

The G4 failure is expected and useful: the engine did not convert strong genome/morphology prediction into strong G2M-RF support when the matched-dimensional PCA control was essentially as accurate as the MGC representation.

This verification demonstrates executable software behavior only. It is not empirical validation of the biological theory.
