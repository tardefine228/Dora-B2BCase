"""Download a SentenceTransformers model into a local directory (Docker build step).

Usage:  python scripts/download_model.py intfloat/multilingual-e5-small /opt/models/embedder

The directory is later passed as ``MODEL_NAME`` with ``MODEL_LOCAL_ONLY=true``, so the
running containers never need internet access.

Only the files a SentenceTransformers bi-encoder needs at load time are kept: ONNX,
OpenVINO and TensorFlow duplicates are skipped. ``pytorch_model.bin`` is skipped as
well **only when the repository also ships ``model.safetensors``** — some models
(``BAAI/bge-m3``) publish the encoder weights exclusively as ``pytorch_model.bin``,
and dropping it used to produce a directory with tokenizer files but no weights:
the image built fine and the runtime silently fell back to spaCy vectors.
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

# Duplicates of the encoder weights we never need in the image.
IGNORE_PATTERNS = [
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "tf_model.h5",
    "rust_model.ot",
]

# Files that actually carry the encoder weights (single-file or sharded layouts).
_ENCODER_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

# Ignore rules that hide the encoder weights and are therefore conditional.
_TORCH_BIN_PATTERNS = ("pytorch_model.bin", "pytorch_model-*.bin")


def is_encoder_weight(name: str) -> bool:
    """True for the file names that make a SentenceTransformers model loadable."""
    base = name.rsplit("/", 1)[-1]
    if base in _ENCODER_WEIGHT_FILES:
        return True
    # Sharded variants: model-00001-of-00002.safetensors, pytorch_model-00001-of-00002.bin.
    return (base.startswith("model-") and base.endswith(".safetensors")) or (
        base.startswith("pytorch_model-") and base.endswith(".bin")
    )


def ignore_patterns_for(file_names: Sequence[str]) -> list[str]:
    """Drop the redundant ``pytorch_model.bin`` only if safetensors weights exist.

    ``scripts/download_model.py`` runs in CI and inside the Docker build, so the
    decision must be made from the repository file list fetched from the HF API,
    not from a hardcoded preference.
    """
    patterns = list(IGNORE_PATTERNS)
    if any(name.endswith(".safetensors") for name in file_names):
        return patterns
    torch_bin = [
        name
        for name in file_names
        if name in _TORCH_BIN_PATTERNS
        or (name.startswith("pytorch_model-") and name.endswith(".bin"))
    ]
    if torch_bin:
        # The only weight files in the repository: keep them.
        patterns = [pattern for pattern in patterns if pattern not in _TORCH_BIN_PATTERNS]
    return patterns


def encoder_weights(directory: Path) -> list[str]:
    """Relative names of the weight files actually present in ``directory``."""
    return sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and is_encoder_weight(path.name)
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit(__doc__)
    repo_id, destination = args
    target = Path(destination)

    from huggingface_hub import HfApi, snapshot_download

    try:
        info = HfApi().model_info(repo_id)
        file_names = [sibling.rfilename for sibling in info.siblings]
    except Exception as exc:  # noqa: BLE001 - the message must reach the build log
        raise SystemExit(f"Не удалось получить список файлов модели {repo_id}: {exc}") from exc

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        ignore_patterns=ignore_patterns_for(file_names),
    )

    weights = encoder_weights(target)
    if not weights:
        present = sorted(path.name for path in target.iterdir()) if target.is_dir() else []
        raise SystemExit(
            f"В {target} нет весов модели {repo_id} (скачано: {present}). "
            "Без весов SentenceTransformer не загрузится и сервис молча откатится "
            "на spaCy — проверьте ignore_patterns в scripts/download_model.py."
        )

    # Preserve identity when Docker copies the model to /opt/models/embedder.
    # The directory name is not a reliable way to select input prefixes.
    (target / "dora_model.json").write_text(
        json.dumps({"model_id": repo_id}, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Model {repo_id} saved to {target} (weights: {', '.join(weights)})")


if __name__ == "__main__":
    main()
