"""Small/medium VCF -> dosage NPZ adapter.

Install: pip install -e .[vcf]
This adapter is intentionally not a national-scale WGS engine; use distributed storage for very large cohorts.
"""
from __future__ import annotations

import argparse

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Convert biallelic VCF genotypes to a dosage NPZ matrix")
    ap.add_argument("vcf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-variants", type=int)
    return ap


def diploid_gt_to_dosage(genotypes: np.ndarray) -> np.ndarray:
    gt = np.asarray(genotypes, dtype=int)
    if gt.ndim != 2 or gt.shape[1] < 2:
        raise ValueError("genotypes must contain at least two allele columns")
    alleles = gt[:, :2]
    dosage = alleles.sum(axis=1).astype(float)
    dosage[np.any(alleles < 0, axis=1)] = np.nan
    return dosage


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_variants is not None and args.max_variants <= 0:
        raise SystemExit("--max-variants must be positive")

    try:
        from cyvcf2 import VCF
    except ImportError as e:
        raise SystemExit("Install optional dependency: pip install -e .[vcf]") from e

    v = VCF(args.vcf)
    samples = np.asarray(v.samples)
    columns = []
    variant_ids = []
    for i, rec in enumerate(v):
        if args.max_variants is not None and i >= args.max_variants:
            break
        if len(rec.ALT) != 1:
            continue
        dosage = diploid_gt_to_dosage(np.asarray(rec.genotypes, dtype=int))
        columns.append(dosage)
        variant_ids.append(f"{rec.CHROM}:{rec.POS}:{rec.REF}:{rec.ALT[0]}")
    if not columns:
        raise SystemExit("No biallelic variants found")
    G = np.column_stack(columns)
    np.savez_compressed(
        args.out,
        sample_ids=samples,
        variant_ids=np.asarray(variant_ids),
        G=G,
    )
    print(f"wrote {G.shape[0]} samples x {G.shape[1]} variants to {args.out}")


if __name__ == "__main__":
    main()
