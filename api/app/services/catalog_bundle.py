"""Bound catalog/source updates. A manifest prevents cross-edition source excerpts."""

import hashlib
import io
import json
import re
import zipfile
from urllib.parse import urlsplit

from app.services.catalog_io import _atomic_write, parse_catalog_bytes

MANIFEST = "source_manifest.json"


def unpack_catalog(filename, content):
    if not filename.lower().endswith(".zip"):
        return parse_catalog_bytes(filename, content), None, {}
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = archive.namelist()
            allowed = {"data.csv", "data.json", "source_text.txt", "metadata.json"}
            if len(set(names)) != len(names) or set(names) - allowed:
                raise ValueError(
                    "ZIP должен содержать только data.csv или data.json, "
                    "source_text.txt и metadata.json в корне."
                )
            if sum(f.file_size for f in archive.infolist()) > 20_000_000:
                raise ValueError("Распакованный каталог превышает 20 МБ.")
            sources = [n for n in ("data.csv", "data.json") if n in names]
            if len(sources) != 1 or "source_text.txt" not in names:
                raise ValueError("Нужны один каталог и source_text.txt.")
            source = archive.read("source_text.txt")
            pages = {
                int(page)
                for page in re.findall(
                    r"^===+\s*СТРАНИЦА\s+(\d+)\s*===+\s*$",
                    source.decode("utf-8"),
                    re.MULTILINE | re.IGNORECASE,
                )
            }
            if not pages:
                raise ValueError("В source_text.txt отсутствуют маркеры страниц.")
            metadata = json.loads(archive.read("metadata.json")) if "metadata.json" in names else {}
            if not isinstance(metadata, dict):
                raise ValueError("metadata.json должен содержать объект.")
            parsed = parse_catalog_bytes(sources[0], archive.read(sources[0]))
            for row in parsed.rows:
                if row.get("code_role") in {"Исключение", "Не указан"}:
                    continue
                try:
                    start = int(row.get("pdf_page_start") or 0)
                    end = int(row.get("pdf_page_end") or start)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Укажите корректные страницы для всех позиций каталога."
                    ) from exc
                if (
                    not (1 <= start <= end <= start + 200)
                    or not set(range(start, end + 1)) <= pages
                ):
                    raise ValueError("В ZIP отсутствуют страницы, указанные в каталоге.")
            return parsed, source, metadata
    except (zipfile.BadZipFile, UnicodeDecodeError, KeyError) as exc:
        raise ValueError("Повреждённый ZIP или текст источника.") from exc


def bind_source(data_dir, source, metadata):
    manifest = {"available": source is not None}
    if source is not None:
        _atomic_write(data_dir / "source_text.txt", source)
        manifest.update(
            {
                "catalog_sha256": hashlib.sha256((data_dir / "data.csv").read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "edition": str(metadata.get("edition", "Редакция не указана")),
                "source_url": str(metadata.get("source_url", "")),
            }
        )
    _atomic_write(data_dir / MANIFEST, json.dumps(manifest, ensure_ascii=False).encode())


def source_is_current(data_dir):
    path = data_dir / MANIFEST
    if not path.exists():
        return True  # Original checked-in snapshot, not yet replaced.
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return manifest.get("available", False) and all(
            hashlib.sha256((data_dir / filename).read_bytes()).hexdigest() == manifest.get(key)
            for filename, key in (
                ("data.csv", "catalog_sha256"),
                ("source_text.txt", "source_sha256"),
            )
        )
    except (OSError, ValueError):
        return False


def source_metadata(data_dir):
    """Unverified edition label and optional safe link supplied by the operator."""
    try:
        metadata = json.loads((data_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "Редакция исходного файла не подтверждена", None
    url = str(metadata.get("source_url", ""))
    try:
        parsed = urlsplit(url)
        safe = parsed.scheme in {"https", "http"} and bool(parsed.netloc)
    except ValueError:
        safe = False
    return str(metadata.get("edition", "Редакция не указана")), url if safe else None
