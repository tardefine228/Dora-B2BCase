"""Weights must survive the snapshot filter for every supported embedder.

``scripts/download_model.py`` used to ignore ``pytorch_model.bin`` unconditionally
to keep the image small. That is correct for ``intfloat/multilingual-e5-small``
(it also ships ``model.safetensors``) and silently fatal for ``BAAI/bge-m3``,
which publishes the encoder weights **only** as ``pytorch_model.bin``: the build
succeeded, the model directory contained tokenizer files, and the runtime fell
back to spaCy vectors without a single error.

The tests stub ``huggingface_hub``: no network and no multi-gigabyte download.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_model.py"

E5_FILES = [
    "1_Pooling/config.json",
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

# Actual file list of BAAI/bge-m3 (HF API): no safetensors at all.
BGE_M3_FILES = [
    "1_Pooling/config.json",
    "README.md",
    "colbert_linear.pt",
    "config.json",
    "config_sentence_transformers.json",
    "long.jpg",
    "modules.json",
    "onnx/model.onnx",
    "onnx/model.onnx_data",
    "onnx/tokenizer.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "sparse_linear.pt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


@pytest.fixture(scope="module")
def downloader():
    spec = importlib.util.spec_from_file_location("dora_download_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_hub(monkeypatch, file_names, written, error: Exception | None = None):
    """Install a stub ``huggingface_hub`` into ``sys.modules``."""
    calls: dict[str, object] = {}

    class HfApi:
        def model_info(self, repo_id):
            calls["repo_id"] = repo_id
            if error is not None:
                raise error
            return types.SimpleNamespace(
                siblings=[types.SimpleNamespace(rfilename=name) for name in file_names]
            )

    def snapshot_download(*, repo_id, local_dir, ignore_patterns):
        calls["ignore_patterns"] = list(ignore_patterns)
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name in written:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("stub", encoding="utf-8")

    module = types.ModuleType("huggingface_hub")
    module.HfApi = HfApi
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def test_encoder_weight_names_cover_single_file_and_sharded_layouts(downloader):
    assert downloader.is_encoder_weight("model.safetensors")
    assert downloader.is_encoder_weight("pytorch_model.bin")
    assert downloader.is_encoder_weight("model-00001-of-00002.safetensors")
    assert downloader.is_encoder_weight("pytorch_model-00001-of-00002.bin")
    assert not downloader.is_encoder_weight("colbert_linear.pt")  # sparse/multi-vector head
    assert not downloader.is_encoder_weight("onnx/model.onnx")


def test_bin_is_kept_when_the_repository_has_no_safetensors(downloader):
    patterns = downloader.ignore_patterns_for(BGE_M3_FILES)
    assert "pytorch_model.bin" not in patterns
    assert "onnx/*" in patterns and "*.onnx" in patterns


def test_bin_is_skipped_when_safetensors_exist(downloader):
    assert "pytorch_model.bin" in downloader.ignore_patterns_for(E5_FILES)


def test_bge_m3_download_keeps_weights_and_records_identity(downloader, monkeypatch, tmp_path):
    # What the (stubbed) snapshot actually writes to disk after filtering.
    written = ["config.json", "modules.json", "tokenizer.json", "pytorch_model.bin"]
    calls = _fake_hub(monkeypatch, BGE_M3_FILES, written)
    target = tmp_path / "embedder"

    downloader.main(["BAAI/bge-m3", str(target)])

    assert calls["repo_id"] == "BAAI/bge-m3"
    assert "pytorch_model.bin" not in calls["ignore_patterns"]
    assert downloader.encoder_weights(target) == ["pytorch_model.bin"]
    assert json.loads((target / "dora_model.json").read_text(encoding="utf-8")) == {
        "model_id": "BAAI/bge-m3"
    }


def test_missing_weights_abort_the_build_instead_of_degrading(downloader, monkeypatch, tmp_path):
    _fake_hub(monkeypatch, BGE_M3_FILES, ["config.json", "tokenizer.json"])
    target = tmp_path / "embedder"

    with pytest.raises(SystemExit) as excinfo:
        downloader.main(["BAAI/bge-m3", str(target)])

    message = str(excinfo.value)
    assert "нет весов" in message
    assert "spaCy" in message
    assert not (target / "dora_model.json").exists()


def test_unreachable_hub_is_reported_with_the_repo_id(downloader, monkeypatch, tmp_path):
    _fake_hub(monkeypatch, [], [], error=RuntimeError("401 Unauthorized"))

    with pytest.raises(SystemExit) as excinfo:
        downloader.main(["closed/model", str(tmp_path / "embedder")])

    assert "closed/model" in str(excinfo.value)


def test_usage_is_printed_for_a_wrong_argument_count(downloader):
    with pytest.raises(SystemExit) as excinfo:
        downloader.main(["BAAI/bge-m3"])
    assert "Usage:" in str(excinfo.value)
