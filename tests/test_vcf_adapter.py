import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vcf_to_genotype_npz.py"


def _load_adapter():
    spec = importlib.util.spec_from_file_location("vcf_adapter", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vcf_adapter_help_works_without_optional_dependency():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--max-variants" in proc.stdout


def test_diploid_dosage_handles_missing_alleles():
    adapter = _load_adapter()
    gt = np.array([[0, 0, False], [0, 1, False], [1, 1, False], [-1, -1, False]])
    dosage = adapter.diploid_gt_to_dosage(gt)
    assert np.allclose(dosage[:3], [0.0, 1.0, 2.0])
    assert np.isnan(dosage[3])


def test_diploid_dosage_rejects_malformed_input():
    adapter = _load_adapter()
    with pytest.raises(ValueError, match="two allele"):
        adapter.diploid_gt_to_dosage(np.ones((4, 1), dtype=int))


def test_vcf_adapter_rejects_nonpositive_max_variants_before_optional_import():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "dummy.vcf", "--out", "x.npz", "--max-variants", "0"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "--max-variants must be positive" in (proc.stdout + proc.stderr)
