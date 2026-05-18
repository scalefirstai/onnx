"""
scalefirst_onnx — minimal loader for models hosted at github.com/scalefirstai/onnx.

Resolves a model name to a pinned GitHub Release, downloads assets to a local
cache, verifies SHA-256 against the bundled manifest, and returns an
onnxruntime InferenceSession plus tokenizer.

No HuggingFace dependency. No network calls after first download.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger("scalefirst_onnx")

GITHUB_REPO = "scalefirstai/onnx"
DEFAULT_CACHE = Path(os.environ.get(
    "SCALEFIRST_ONNX_CACHE",
    Path.home() / ".cache" / "scalefirst" / "onnx",
))

# Internal mirror override — set this in regulated/air-gapped environments to
# point at Artifactory / internal S3 instead of GitHub.
MIRROR_BASE = os.environ.get("SCALEFIRST_ONNX_MIRROR")  # e.g. https://artifactory.example.com/onnx


# Registry of supported models. Versions are pinned here, not resolved dynamically,
# so consumers get reproducible builds.
REGISTRY: dict[str, dict[str, Any]] = {
    "all-MiniLM-L6-v2": {
        "version": "v1.0.0",
        "tag": "all-MiniLM-L6-v2-v1.0.0",
        "files": {
            "fp32": "model.onnx",
            "int8": "model.quant.onnx",
            "tokenizer": "tokenizer.json",
            "vocab": "vocab.txt",
            "config": "config.json",
            "tokenizer_config": "tokenizer_config.json",
            "special_tokens": "special_tokens_map.json",
            "manifest": "manifest.json",
        },
    },
}


@dataclass
class ModelPaths:
    root: Path
    model: Path
    tokenizer: Path
    manifest: dict


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _asset_url(tag: str, filename: str) -> str:
    if MIRROR_BASE:
        return f"{MIRROR_BASE.rstrip('/')}/{tag}/{filename}"
    return f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/{filename}"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    LOG.info("downloading %s", url)
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            out.write(buf)
    tmp.rename(dest)


def _ensure_file(tag: str, filename: str, cache_dir: Path,
                 expected_sha: str | None = None) -> Path:
    dest = cache_dir / filename
    if dest.exists():
        if expected_sha and _sha256(dest) != expected_sha:
            LOG.warning("checksum mismatch on cached %s, re-downloading", filename)
            dest.unlink()
        else:
            return dest
    _download(_asset_url(tag, filename), dest)
    if expected_sha:
        got = _sha256(dest)
        if got != expected_sha:
            dest.unlink()
            raise RuntimeError(
                f"sha256 mismatch for {filename}: expected {expected_sha}, got {got}"
            )
    return dest


def fetch(model_name: str, quantized: bool = True,
          cache_dir: Path | None = None) -> ModelPaths:
    """Download (if needed), verify, and return paths to model files."""
    if model_name not in REGISTRY:
        raise KeyError(f"unknown model: {model_name}. Known: {list(REGISTRY)}")
    entry = REGISTRY[model_name]
    tag = entry["tag"]
    files = entry["files"]

    cache_dir = (cache_dir or DEFAULT_CACHE) / model_name / entry["version"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Fetch manifest first (unchecked — it carries the checksums for everything else).
    manifest_path = _ensure_file(tag, files["manifest"], cache_dir)
    manifest = json.loads(manifest_path.read_text())
    sha_by_name = {f["name"]: f["sha256"] for f in manifest["files"]}

    model_file = files["int8"] if quantized else files["fp32"]
    needed = [
        model_file,
        files["tokenizer"], files["vocab"],
        files["config"], files["tokenizer_config"], files["special_tokens"],
    ]
    for name in needed:
        _ensure_file(tag, name, cache_dir, expected_sha=sha_by_name.get(name))

    return ModelPaths(
        root=cache_dir,
        model=cache_dir / model_file,
        tokenizer=cache_dir / files["tokenizer"],
        manifest=manifest,
    )


def load_model(model_name: str, quantized: bool = True,
               providers: list[str] | None = None,
               cache_dir: Path | None = None):
    """Return (onnxruntime.InferenceSession, tokenizers.Tokenizer)."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    paths = fetch(model_name, quantized=quantized, cache_dir=cache_dir)
    sess = ort.InferenceSession(
        str(paths.model),
        providers=providers or ["CPUExecutionProvider"],
    )
    tok = Tokenizer.from_file(str(paths.tokenizer))
    return sess, tok


def encode(model_name: str, texts: list[str], quantized: bool = True) -> "Any":
    """Convenience: run the model and return mean-pooled, L2-normalized embeddings."""
    import numpy as np

    sess, tok = load_model(model_name, quantized=quantized)
    tok.enable_padding()
    tok.enable_truncation(max_length=256)
    encs = tok.encode_batch(texts)

    input_ids = np.array([e.ids for e in encs], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    outputs = sess.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })
    # Mean-pool token embeddings weighted by attention mask.
    last_hidden = outputs[0]  # (batch, seq, hidden)
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    pooled = summed / counts
    # L2 normalize.
    norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
    return pooled / norms
