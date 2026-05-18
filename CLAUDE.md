# Instructions for Claude Code

This repository hosts ONNX model exports as GitHub Release assets for environments
where HuggingFace Hub is not reachable.

## Project layout

```
scripts/publish_model.py        Pulls from HF, exports ONNX, quantizes, builds manifest
scalefirst_onnx/__init__.py     Consumer-side loader (no HF dependency)
.github/workflows/publish.yml   Manual-trigger workflow that creates a release
docs/deployment.md              Internal mirror pattern for regulated environments
pyproject.toml                  Loader package metadata
```

## Bootstrap tasks

When asked to "bootstrap" or "set up" this repo for the first time:

1. Create the GitHub repo `scalefirstai/onnx` (gh repo create scalefirstai/onnx --public)
2. Push this tree
3. Trigger the publish workflow for `all-MiniLM-L6-v2`:
   ```
   gh workflow run publish.yml \
     -f model=sentence-transformers/all-MiniLM-L6-v2 \
     -f version=v1.0.0
   ```
4. Wait for the release to appear, then verify with the loader:
   ```
   pip install -e .
   python -c "from scalefirst_onnx import encode; print(encode('all-MiniLM-L6-v2', ['hello world']).shape)"
   ```

## Adding a new model

1. Add an entry to `REGISTRY` in `scalefirst_onnx/__init__.py` (pin a version+tag).
2. Trigger the workflow with the new HF model id and version.
3. Update README.md model table.

## Conventions

- Release tags: `<short-model-name>-v<semver>` (e.g. `all-MiniLM-L6-v2-v1.0.0`).
- Never overwrite an existing tag. Bump the version instead.
- Every release must include `manifest.json`, `SOURCE.md`, and `LICENSE`.
- Quantized variant is named `model.quant.onnx`, full-precision is `model.onnx`.
- Opset 17 unless a specific model needs otherwise.

## Out of scope

- Do not add runtime HF Hub calls anywhere. The whole point is no HF dependency.
- Do not use Git LFS. Releases are sized for direct release-asset hosting.
- Do not auto-update model versions. Consumers depend on pinned tags.
