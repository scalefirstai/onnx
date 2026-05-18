# scalefirstai/onnx

GitHub-hosted ONNX models for environments where HuggingFace Hub is not reachable
(corporate proxies, regulated networks, air-gapped CI).

Models are published as **GitHub Release assets** with SHA-256 manifests so consumers
can verify integrity. All models retain their original upstream licenses; this repo
only redistributes weights and tokenizer assets unchanged.

## Available models

| Model | Size (fp32) | Size (int8) | License | Upstream |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | ~90 MB | ~23 MB | Apache-2.0 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

## Quick start

```python
from scalefirst_onnx import load_model

session, tokenizer = load_model("all-MiniLM-L6-v2", quantized=True)
```

The loader resolves the cache path (env var `SCALEFIRST_ONNX_CACHE`, default
`~/.cache/scalefirst/onnx`), downloads from the pinned release tag if missing,
and verifies SHA-256 against the bundled manifest.

## Release layout

Each model is published under a tag like `all-MiniLM-L6-v2-v1.0.0`. Assets:

```
model.onnx              # fp32 graph
model.quant.onnx        # int8 quantized graph
tokenizer.json          # fast tokenizer
vocab.txt               # wordpiece vocab
config.json             # model config
tokenizer_config.json   # tokenizer config
special_tokens_map.json
manifest.json           # filenames + sha256 + sizes
LICENSE                 # upstream license copy
SOURCE.md               # upstream commit hash + provenance
```

## For regulated / corporate environments

Recommended deployment pattern:

1. CI fetches a pinned release from `github.com/scalefirstai/onnx` once
2. Re-publishes to internal Artifactory / S3 with the same manifest
3. Application code reads from internal storage at runtime — no external dependency

See [`docs/deployment.md`](docs/deployment.md) for the mirror pattern.
