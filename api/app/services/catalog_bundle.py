"""Bounded ZIP ingestion and immutable publication of complete decree editions."""
import io
import json
import re
import uuid
import zipfile
from pathlib import Path

from ml.catalog_revision import active_catalog_dir

FILES = ("data.csv", "data.json", "source_text.txt", "report.json")


def parse_bundle(content):
    from app.services.catalog_io import parse_catalog_bytes

    if len(content) > 5_000_000:
        raise ValueError("Пакет превышает 5 МБ")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            required = {"data.csv", "source_text.txt", "report.json"}
            if len(entries) != 3 or {e.filename for e in entries} != required:
                raise ValueError(
                    "ZIP: нужны ровно data.csv, source_text.txt, report.json"
                )
            if sum(e.file_size for e in entries) > 20_000_000:
                raise ValueError("Распакованный пакет превышает 20 МБ")
            blobs = {e.filename: archive.read(e) for e in entries}
        parsed = parse_catalog_bytes("data.csv", blobs["data.csv"])
        text = blobs["source_text.txt"].decode("utf-8-sig")
        report = json.loads(blobs["report.json"].decode("utf-8-sig"))
        if not isinstance(report, dict):
            raise ValueError("report.json должен быть объектом")
        markers = re.findall(r"^===+\s*СТРАНИЦА\s+(\d+)\s*===+\s*$", text, re.M | re.I)
        pages = {int(p) for p in markers}
        if not pages or len(pages) != len(markers):
            raise ValueError("Нет страниц первоисточника или повторяются номера страниц")
        ids = set()
        for row in parsed.rows:
            rid = row.get("record_id")
            start, end = int(row["pdf_page_start"]), int(row["pdf_page_end"])
            if not rid or rid in ids or not 1 <= start <= end or end - start > len(pages):
                raise ValueError("Некорректный record_id или диапазон страниц")
            if not all(page in pages for page in range(start, end + 1)):
                raise ValueError(f"Нет страниц первоисточника для {rid}")
            ids.add(rid)
        parsed.source_files = {k: blobs[k] for k in ("source_text.txt", "report.json")}
        return parsed
    except (
        zipfile.BadZipFile, UnicodeError, KeyError, TypeError, RuntimeError, NotImplementedError
    ) as exc:
        raise ValueError("Некорректный пакет каталога") from exc


def publish(root: Path, blobs: dict[str, bytes]):
    from app.services.catalog_io import _atomic_write

    version = uuid.uuid4().hex
    target = root / "revisions" / version
    target.mkdir(parents=True)
    for name, content in blobs.items():
        if name not in FILES:
            raise ValueError("Unknown revision file")
        _atomic_write(target / name, content)
    # Readers see either complete old or complete new files, never a partial edition.
    _atomic_write(root / "active-catalog.json", json.dumps({"revision": version}).encode())


def current_files(root):
    current = active_catalog_dir(root)
    return {name: (current / name).read_bytes() for name in FILES if (current / name).is_file()}


def serialized_update(function):
    """Serialize API and worker writers using a lock on the shared data volume."""
    import inspect
    from functools import wraps

    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        root = Path(bound.arguments["data_dir"])
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".catalog-update.lock").open("a+b") as lock:
            import os
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                lock.write(b"0")
                lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return function(*args, **kwargs)
            finally:
                if os.name == "nt":
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return wrapped
