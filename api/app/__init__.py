"""API application package.

The package keeps the project root (the directory holding both ``app/`` and the
sibling ``ml/`` package) importable. The API process gets that directory on
``sys.path`` from uvicorn (its ``app_dir`` defaults to the working directory),
but the Celery entrypoint (``celery -A app.worker.celery_app``) does not: there
``app.*`` resolves while the lazily imported ``ml`` package stays invisible,
which used to kill every batch job with ``ModuleNotFoundError: No module named
'ml'``. Adding the root here makes every entrypoint see both packages.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_root_on_syspath() -> None:
    """Append the directory that contains both ``app/`` and ``ml/``.

    Walks up from this file so it works in the image (``/app/app/__init__.py``
    -> ``/app``) and in the repository (``api/app/__init__.py`` -> repo root).
    Appending (not inserting) keeps site-packages priority untouched; when no
    ``ml`` package exists above us (installed-wheel scenario) nothing happens.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "ml" / "__init__.py").is_file():
            root = str(parent)
            if root not in sys.path:
                sys.path.append(root)
            return


_ensure_project_root_on_syspath()
