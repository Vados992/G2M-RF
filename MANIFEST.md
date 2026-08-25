# Repository manifest

- `README.md` — installation, execution and scientific boundaries
- `pyproject.toml` — Python package definition and CLI entry point
- `src/g2mrf/geometry` — M0–M5 equations, physical envelopes, fitting and transforms
- `src/g2mrf/genomics` — TRAIN-only standardization, kernels, corrected FWL, exact/scalable KRR
- `src/g2mrf/statistics` — metrics, bootstrap and Holm correction
- `src/g2mrf/data` — analysis-ready schema and synthetic generator
- `src/g2mrf/pipeline.py` — complete confirmatory execution pipeline
- `src/g2mrf/gates.py` — G1–G5 decision engine
- `src/g2mrf/planning.py` — sample-size planning envelope
- `src/g2mrf/cli.py` — `g2mrf` command-line interface
- `tests/` — 20 mathematical/statistical/software tests
- `configs/default.yaml` — frozen example thresholds and model settings
- `docs/` — architecture, data schema, real-data checklist, validation protocol and canonical theory-document manifest
- `scripts/vcf_to_genotype_npz.py` — optional VCF adapter for moderate datasets
- `scripts/publish_to_github.ps1` — Windows Git push helper
- `.github/workflows/ci.yml` — Python 3.10–3.12 CI
- `Dockerfile` — reproducible runtime container
- `examples/demo_report.json` — verified synthetic run output
