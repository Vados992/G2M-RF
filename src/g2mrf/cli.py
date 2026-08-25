from __future__ import annotations

import argparse
import json

from .config import RunConfig
from .data.synthetic import make_synthetic_dataset
from .data.io import load_npz, save_npz
from .pipeline import run_confirmatory, write_report
from .planning import planning_grid, required_n


def cmd_demo(args):
    cfg = RunConfig.from_yaml(args.config) if args.config else RunConfig()
    if args.bootstrap is not None:
        cfg.model.bootstrap = args.bootstrap
    bundle = make_synthetic_dataset(args.n, args.variants, args.landmarks, cfg.model.seed)
    report = run_confirmatory(bundle, cfg)
    out = write_report(report, args.out)
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    print(f"report: {out}")


def cmd_plan(args):
    rows = planning_grid(tuple(args.n), args.h2, args.me)
    print("N_TRAIN\tR2_pred")
    for row in rows:
        print(f"{row['n_train']}\t{row['r2_pred']:.6f}")
    if args.target is not None:
        print(f"required N for R2={args.target}: {required_n(args.target, args.h2, args.me):.0f}")


def cmd_generate(args):
    bundle = make_synthetic_dataset(args.n, args.variants, args.landmarks, args.seed)
    save_npz(bundle, args.out)
    print(args.out)


def cmd_run(args):
    cfg = RunConfig.from_yaml(args.config) if args.config else RunConfig()
    report = run_confirmatory(load_npz(args.data), cfg)
    write_report(report, args.out)
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


def build_parser():
    p = argparse.ArgumentParser(prog="g2mrf", description="G2M-RF executable research framework")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run a complete synthetic train/internal/external validation")
    d.add_argument("--n", type=int, default=720)
    d.add_argument("--variants", type=int, default=500)
    d.add_argument("--landmarks", type=int, default=25)
    d.add_argument("--bootstrap", type=int, default=None)
    d.add_argument("--config")
    d.add_argument("--out", default="results/demo_report.json")
    d.set_defaults(func=cmd_demo)

    q = sub.add_parser("plan", help="Daetwyler-style planning envelope; never a gate result")
    q.add_argument("--h2", type=float, default=0.35)
    q.add_argument("--me", type=float, default=75000)
    q.add_argument("--n", type=int, nargs="+", default=[20000, 50000, 100000, 200000, 500000])
    q.add_argument("--target", type=float)
    q.set_defaults(func=cmd_plan)

    g = sub.add_parser("generate", help="generate an analysis-ready synthetic NPZ dataset")
    g.add_argument("--n", type=int, default=720)
    g.add_argument("--variants", type=int, default=500)
    g.add_argument("--landmarks", type=int, default=25)
    g.add_argument("--seed", type=int, default=1601001)
    g.add_argument("--out", default="examples/synthetic.npz")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="run the frozen pipeline on an analysis-ready NPZ bundle")
    r.add_argument("--data", required=True)
    r.add_argument("--config")
    r.add_argument("--out", default="results/report.json")
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
