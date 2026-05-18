# Deployment for regulated / corporate environments

## The mirror pattern

Don't pull from GitHub at runtime in production. Pull once, store internally,
point applications at internal storage. This gives you:

- Reproducible builds (pinned by SHA-256, not by what's on github.com today)
- No external runtime dependency
- Audit trail through your existing artifact governance
- Survives GitHub outages and network policy changes

### One-time mirror sync

On a machine with outbound GitHub access (typically a CI runner or jump host):

```bash
TAG="all-MiniLM-L6-v2-v1.0.0"
BASE="https://github.com/scalefirstai/onnx/releases/download/${TAG}"

mkdir -p mirror/${TAG} && cd mirror/${TAG}
for f in model.onnx model.quant.onnx tokenizer.json vocab.txt \
         config.json tokenizer_config.json special_tokens_map.json \
         manifest.json SOURCE.md LICENSE; do
  curl -fSL -O "${BASE}/${f}"
done

# Verify checksums against the manifest
python -c "
import json, hashlib, sys
m = json.load(open('manifest.json'))
for f in m['files']:
    h = hashlib.sha256(open(f['name'],'rb').read()).hexdigest()
    assert h == f['sha256'], f['name']
    print('ok', f['name'])
"
```

Then upload `mirror/${TAG}/` to your internal store:

- **Artifactory generic repo**: `jfrog rt u "mirror/${TAG}/*" generic-local/onnx/${TAG}/`
- **S3 bucket**: `aws s3 sync mirror/${TAG}/ s3://internal-models/onnx/${TAG}/`
- **Nexus raw repo**: `curl --upload-file` per file

### Application configuration

Set the mirror base in your runtime environment:

```bash
export SCALEFIRST_ONNX_MIRROR="https://artifactory.example.com/generic-local/onnx"
# or
export SCALEFIRST_ONNX_MIRROR="https://internal-models.s3.amazonaws.com/onnx"
```

The loader will fetch from `${SCALEFIRST_ONNX_MIRROR}/${TAG}/${FILENAME}` instead
of GitHub. Checksum verification still runs against the bundled manifest, so a
corrupted or tampered mirror copy will fail loudly.

## PyPI-only environments (no source installs)

In environments where only PyPI-published packages may be installed — and
`scalefirst-onnx` itself isn't on PyPI — you don't need this package at runtime.
The release assets are static files; the loader is convenience, not protocol.

Once the tag is mirrored to your internal store, application code only needs
`onnxruntime`, `tokenizers`, and `numpy` (all on PyPI):

```python
import hashlib, json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path("/opt/models/all-MiniLM-L6-v2")  # populated from your S3 mirror

# Verify checksums against the bundled manifest before trusting the files.
manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
for f in manifest["files"]:
    got = hashlib.sha256((MODEL_DIR / f["name"]).read_bytes()).hexdigest()
    assert got == f["sha256"], f"sha256 mismatch on {f['name']}"

sess = ort.InferenceSession(str(MODEL_DIR / "model.quant.onnx"),
                            providers=["CPUExecutionProvider"])
tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
tok.enable_padding(); tok.enable_truncation(max_length=256)


def embed(texts: list[str]) -> np.ndarray:
    encs = tok.encode_batch(texts)
    ids  = np.array([e.ids            for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    last_hidden = sess.run(None, {
        "input_ids": ids,
        "attention_mask": mask,
        "token_type_ids": np.zeros_like(ids),
    })[0]
    # mean-pool weighted by attention mask, then L2-normalize
    m = mask[..., None].astype(np.float32)
    pooled = (last_hidden * m).sum(1) / m.sum(1).clip(min=1e-9)
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
```

If you prefer the `load_model(...)` ergonomics without installing this package,
vendor `scalefirst_onnx/__init__.py` into your codebase as a single file —
its only runtime deps are `onnxruntime`, `tokenizers`, and `numpy`, all on PyPI.
Set `SCALEFIRST_ONNX_MIRROR` to your S3 base URL and it will fetch + verify
exactly as the installed package would.

## Container builds

For containerized services, bake the model into the image at build time:

```dockerfile
FROM python:3.11-slim AS model-fetch
ARG MODEL_TAG=all-MiniLM-L6-v2-v1.0.0
ENV SCALEFIRST_ONNX_CACHE=/opt/models
RUN pip install --no-cache-dir scalefirst-onnx
RUN python -c "from scalefirst_onnx import fetch; fetch('all-MiniLM-L6-v2')"

FROM python:3.11-slim
COPY --from=model-fetch /opt/models /opt/models
ENV SCALEFIRST_ONNX_CACHE=/opt/models
# ... your app
```

This produces an image with the model baked in and zero network calls at startup.

## Verification at load time

The loader verifies SHA-256 on every file against the manifest before returning
the session. If you want belt-and-braces, also verify the manifest itself against
a known-good hash stored in your config:

```python
import hashlib
from scalefirst_onnx import fetch

paths = fetch("all-MiniLM-L6-v2")
manifest_sha = hashlib.sha256((paths.root / "manifest.json").read_bytes()).hexdigest()
assert manifest_sha == EXPECTED_MANIFEST_SHA, "manifest tampered"
```

## License & provenance

Every release ships with:

- `LICENSE` — copy of the upstream license (Apache-2.0 for all-MiniLM-L6-v2)
- `SOURCE.md` — upstream repo, commit hash, export date, exporter version

This is your audit evidence that the redistributed artifact matches a specific
upstream snapshot.
