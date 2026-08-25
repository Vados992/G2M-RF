# Changelog

## 2.0.0 — 2026-08-25

Initial executable implementation of the corrected G2M-RF v2.0 research framework.

- implemented M0–M5 geometry families;
- implemented corrected penalized-FWL handling;
- implemented fixed-effect KRR and scalable Nyström backend;
- implemented direct/MGC/PCA branches;
- implemented G1–G5 engine and bootstrap/CI logic;
- implemented family-aware deterministic split and leakage firewall;
- added planning envelope, tests, CI, Docker, documentation and synthetic reference run.

### Verification and production hardening

- expanded testing from the initial smoke/unit suite to layered unit, integration, numerical, reproducibility, adversarial, G1–G5 and scaling-regression coverage;
- fixed transferability decision ordering so `NOT TRANSPORTABLE` is reachable and distinct from G5 compression failure;
- added strict analysis-ready `DataBundle` schema validation;
- hardened genotype dosage, family split, kernel, KRR/Nyström/FWL, metric and bootstrap API contracts;
- made report JSON RFC-compatible by converting non-finite floating values to `null`;
- embedded software/runtime provenance in analysis reports;
- added Exact-vs-full-rank-Nyström numerical comparison and an N x N kernel-allocation regression guard;
- upgraded CI to branch-coverage enforcement plus Ubuntu/Windows/macOS compatibility jobs;
- added wheel build, installed CLI end-to-end smoke, Docker build/runtime smoke and coverage artifact generation;
- moved Docker execution to an unprivileged runtime user;
- made the optional VCF adapter expose `--help` without requiring `cyvcf2` and added dosage adapter tests;
- added Dependabot monitoring for Python and GitHub Actions dependencies.
