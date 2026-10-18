"""The Celery worker must see the sibling ``ml`` package (no Docker needed).

Regression: batch jobs died in the worker with
``ModuleNotFoundError: No module named 'ml'`` while the API process was fine.
Both packages live side by side in the image (``/app/app``, ``/app/ml``); uvicorn
puts the working directory on ``sys.path``, the ``celery`` console script does
not, so ``app.*`` resolved there but the lazily imported ``ml`` did not.
``app/__init__`` now adds the project root itself — this test reproduces the
worker's import conditions in a subprocess (cwd stripped from ``sys.path``),
which needs nothing beyond the standard library.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROBE = "\n".join(
    (
        "import os, sys",
        "cwd = os.path.abspath(os.curdir)",
        # The celery console script never adds the working directory; drop it
        # (and the empty entry that means cwd) the same way.
        "sys.path = [p for p in sys.path if os.path.abspath(p or os.curdir) != cwd]",
        "sys.path.insert(0, os.path.join(cwd, 'api'))",
        "import app",
        "import ml.features",
        "print('ml-ok')",
    )
)


def test_ml_package_importable_without_cwd_on_syspath() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "api")}
    env.pop("PYTHONSAFEPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ml-ok" in result.stdout


def test_project_root_bootstrap_is_idempotent() -> None:
    import app  # noqa: F401 - the import itself runs the bootstrap

    roots = [p for p in sys.path if (Path(p) / "ml" / "__init__.py").is_file()]
    assert roots, "project root with the ml package not on sys.path"
    import app as app_again  # noqa: F401

    roots_again = [p for p in sys.path if (Path(p) / "ml" / "__init__.py").is_file()]
    assert roots_again == roots, "bootstrap appended the root twice"
