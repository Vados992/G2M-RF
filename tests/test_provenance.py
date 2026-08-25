from g2mrf import __version__
from g2mrf.provenance import software_provenance


def test_software_provenance_contains_reproducibility_fields():
    p = software_provenance()
    required = {
        "g2mrf",
        "python",
        "python_implementation",
        "platform",
        "numpy",
        "scipy",
        "scikit_learn",
        "pyyaml",
        "git_commit",
    }
    assert set(p) == required
    assert p["g2mrf"] == __version__
    for key in required - {"git_commit"}:
        assert isinstance(p[key], str)
        assert p[key]


def test_github_sha_is_captured_when_present(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    assert software_provenance()["git_commit"] == "abc123"
