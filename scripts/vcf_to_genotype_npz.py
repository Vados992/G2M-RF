"""Small/medium VCF -> dosage NPZ adapter.

Install: pip install -e .[vcf]
This adapter is intentionally not a national-scale WGS engine; use distributed storage for very large cohorts.
"""
from __future__ import annotations

import argparse
import numpy as np


def main():
    try:
        from cyvcf2 import VCF
    except ImportError as e:
        raise SystemExit("Install optional dependency: pip install -e .[vcf]") from e

    ap = argparse.ArgumentParser()
    ap.add_argument("vcf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-variants", type=int)
    args = ap.parse_args()

    v = VCF(args.vcf)
    samples = np.asarray(v.samples)
    columns = []
    variant_ids = []
    for i, rec in enumerate(v):
        if args.max_variants is not None and i >= args.max_variants:
            break
        if len(rec.ALT) != 1:
            continue
        gt = np.asarray(rec.genotypes, dtype=int)[:, :2]
        dosage = gt.sum(axis=1).astype(float)
        dosage[np.any(gt < 0, axis=1)] = np.nan
        columns.append(dosage)
        variant_ids.append(f"{rec.CHROM}:{rec.POS}:{rec.REF}:{rec.ALT[0]}")
    if not columns:
        raise SystemExit("No biallelic variants found")
    G = np.column_stack(columns)
    np.savez_compressed(args.out, sample_ids=samples, variant_ids=np.asarray(variant_ids), G=G)
    print(f"wrote {G.shape[0]} samples x {G.shape[1]} variants to {args.out}")


if __name__ == "__main__":
    main()
