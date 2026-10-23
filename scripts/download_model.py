"""Download a SentenceTransformers model into a local directory (Docker build step).

Usage:  python scripts/download_model.py intfloat/multilingual-e5-small /opt/models/embedder

The directory is later passed as ``MODEL_NAME`` with ``MODEL_LOCAL_ONLY=true``, so the
running containers never need internet access.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    repo_id, destination = sys.argv[1], Path(sys.argv[2])
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        # PyTorch weights only: skip ONNX/OpenVINO/TF duplicates (hundreds of MB).
        ignore_patterns=[
            "onnx/*",
            "openvino/*",
            "*.onnx",
            "*.h5",
            "*.msgpack",
            "*.ot",
            "pytorch_model.bin",
            "tf_model.h5",
            "rust_model.ot",
        ],
    )
    # Preserve identity when Docker copies the model to /opt/models/embedder.
    # The directory name is not a reliable way to select E5 input prefixes.
    (destination / "dora_model.json").write_text(
        json.dumps({"model_id": repo_id}, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Model {repo_id} saved to {destination}")


if __name__ == "__main__":
    main()

