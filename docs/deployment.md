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
- **S3 bucket**: `aws s3 sync mirror/${TAG}/ s3://bny-internal-models/onnx/${TAG}/`
- **Nexus raw repo**: `curl --upload-file` per file

### Application configuration

Set the mirror base in your runtime environment:

```bash
export SCALEFIRST_ONNX_MIRROR="https://artifactory.bny.example/generic-local/onnx"
# or
export SCALEFIRST_ONNX_MIRROR="https://bny-internal-models.s3.amazonaws.com/onnx"
```

The loader will fetch from `${SCALEFIRST_ONNX_MIRROR}/${TAG}/${FILENAME}` instead
of GitHub. Checksum verification still runs against the bundled manifest, so a
corrupted or tampered mirror copy will fail loudly.

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
