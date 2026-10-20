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

# Same, for a layout where ``ml`` has no ``__init__.py`` (namespace package):
# there is no ``api`` subdirectory to insert, PYTHONPATH alone carries ``app``.
PROBE_NAMESPACE = "\n".join(
    (
        "import os, sys",
        "cwd = os.path.abspath(os.curdir)",
        "sys.path = [p for p in sys.path if os.path.abspath(p or os.curdir) != cwd]",
        "import app",
        "import ml.features",
        "assert ml.features.VALUE == 42",
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


def test_bootstrap_recognizes_ml_without_init_file(tmp_path: Path) -> None:
    # A checkout/image that lost ml/__init__.py must still work: the package is
    # then a PEP 420 namespace package once its parent is on sys.path.
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text(
        (ROOT / "api" / "app" / "__init__.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "ml").mkdir()
    (tmp_path / "ml" / "features.py").write_text("VALUE = 42\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    env.pop("PYTHONSAFEPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", PROBE_NAMESPACE],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ml-ok" in result.stdout
