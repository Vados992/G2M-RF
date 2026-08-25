from __future__ import annotations

import os
import platform
from importlib.metadata import PackageNotFoundError, version

from . import __version__


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def software_provenance() -> dict[str, str | None]:
    """Return runtime metadata required to reproduce a reported analysis environment."""
    return {
        "g2mrf": __version__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "scikit_learn": _package_version("scikit-learn"),
        "pyyaml": _package_version("PyYAML"),
        "git_commit": os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT"),
    }
