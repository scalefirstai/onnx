#!/usr/bin/env python3
"""
publish_model.py — fetch a sentence-transformers model, export to ONNX,
quantize, generate manifest, and stage assets for a GitHub Release.

Run this on a machine that has HuggingFace access. The produced
`./release/<model>-<version>/` directory is what gets uploaded as
release assets (via `gh release create` or the workflow in
.github/workflows/publish.yml).

Usage:
    python scripts/publish_model.py \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --version v1.0.0 \\
        --output-dir ./release
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def export_onnx(model_id: str, work_dir: Path) -> None:
    """Use optimum-cli to export to ONNX. Produces model.onnx + tokenizer files."""
    work_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", model_id,
        "--task", "feature-extraction",
        "--opset", "17",
        str(work_dir),
    ])


def quantize_onnx(src: Path, dst: Path) -> None:
    """Dynamic int8 quantization via onnxruntime."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
    )


def resolve_upstream_commit(model_id: str) -> str | None:
    """Best-effort: read the commit hash optimum cached. None if not resolvable."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id)
        return info.sha
    except Exception as e:
        print(f"  ! could not resolve upstream commit: {e}")
        return None


def build_manifest(release_dir: Path, model_id: str, version: str,
                   upstream_sha: str | None) -> dict:
    files = []
    for p in sorted(release_dir.iterdir()):
        if p.name == "manifest.json" or p.is_dir():
            continue
        files.append({
            "name": p.name,
            "size": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    return {
        "model_id": model_id,
        "version": version,
        "upstream": {
            "source": f"https://huggingface.co/{model_id}",
            "commit": upstream_sha,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


def copy_license(work_dir: Path, release_dir: Path) -> None:
    """Copy any LICENSE / README from the HF snapshot into the release."""
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "README.md"):
        src = work_dir / name
        if src.exists():
            shutil.copy2(src, release_dir / name)
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id, e.g. sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--version", required=True, help="Release version, e.g. v1.0.0")
    ap.add_argument("--output-dir", default="./release", type=Path)
    ap.add_argument("--skip-quantize", action="store_true")
    args = ap.parse_args()

    short_name = args.model.split("/")[-1]
    work_dir = args.output_dir / f"_work_{short_name}"
    release_dir = args.output_dir / f"{short_name}-{args.version}"
    release_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Exporting {args.model} to ONNX...")
    export_onnx(args.model, work_dir)

    print(f"[2/5] Copying assets to release dir...")
    keep = {
        "model.onnx", "tokenizer.json", "vocab.txt", "config.json",
        "tokenizer_config.json", "special_tokens_map.json",
    }
    for name in keep:
        src = work_dir / name
        if src.exists():
            shutil.copy2(src, release_dir / name)
        else:
            print(f"  ! missing optional file: {name}")
    copy_license(work_dir, release_dir)

    if not args.skip_quantize:
        print(f"[3/5] Quantizing to int8...")
        quantize_onnx(release_dir / "model.onnx", release_dir / "model.quant.onnx")
    else:
        print(f"[3/5] Skipping quantization")

    print(f"[4/5] Resolving upstream commit...")
    upstream_sha = resolve_upstream_commit(args.model)

    print(f"[5/5] Writing manifest.json + SOURCE.md...")
    manifest = build_manifest(release_dir, args.model, args.version, upstream_sha)
    (release_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (release_dir / "SOURCE.md").write_text(
        f"# Source\n\n"
        f"- Upstream: https://huggingface.co/{args.model}\n"
        f"- Commit: `{upstream_sha or 'unresolved'}`\n"
        f"- Exported: {manifest['created_at']}\n"
        f"- Exporter: optimum.exporters.onnx (opset 17, task=feature-extraction)\n"
    )

    print(f"\nDone. Release staged at: {release_dir}")
    print(f"Total files: {len(manifest['files'])}")
    for f in manifest["files"]:
        print(f"  {f['name']:30s}  {f['size']:>12,}  {f['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
