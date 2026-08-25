# ESE container images

The container build uses the current Expert Streaming Engine source. Images are
published by the manual
[`build-container.yml`](../.github/workflows/build-container.yml) workflow to:

```text
ghcr.io/xero00000/ik-llama-cpp
```

The package name is retained for compatibility with the inherited container
layout. It should not be confused with an upstream `ik_llama.cpp` checkout.

## Variants

Each backend is published with three targets:

| Backend | Server | Full tool image | llama-swap image |
| --- | --- | --- | --- |
| CPU | `cpu-server` | `cpu-full` | `cpu-swap` |
| CUDA 12 | `cu12-server` | `cu12-full` | `cu12-swap` |
| CUDA 13 | `cu13-server` | `cu13-full` | `cu13-swap` |

The workflow also publishes immutable tags with the Git commit-count build
number appended, such as `cu12-server-1234`.

## Run the server

CPU example:

```bash
docker run --rm \
  -p 8080:8080 \
  -v /path/to/models:/models:ro \
  ghcr.io/xero00000/ik-llama-cpp:cpu-server \
  -m /models/model.gguf --host 0.0.0.0 --port 8080
```

CUDA example using the NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all \
  -p 8080:8080 \
  -v /path/to/models:/models:ro \
  ghcr.io/xero00000/ik-llama-cpp:cu12-server \
  -m /models/model.gguf --host 0.0.0.0 --port 8080
```

Check readiness at `http://127.0.0.1:8080/health`.

The `server` target starts the native `llama-server` directly. It does not run
the Python `ese` launcher, so pass native ESE memory-controller options such as
`--memory-policy`, `--max-ram`, and `--reserve-vram` explicitly when needed.

## Full image

The `full` target contains the native binaries plus the repository's Python
conversion and helper tools:

```bash
docker run --rm -it --entrypoint /bin/bash \
  -v /path/to/models:/models \
  ghcr.io/xero00000/ik-llama-cpp:cpu-full
```

## llama-swap image

The `swap` target wraps the server with the checked-in example configuration.
Mount your own configuration when using it outside a local test:

```bash
docker run --rm --gpus all \
  -p 8080:8080 \
  -v /path/to/models:/models:ro \
  -v /path/to/config.yaml:/app/config.yaml:ro \
  ghcr.io/xero00000/ik-llama-cpp:cu12-swap
```

## Build locally

Docker Buildx Bake builds all three targets. Choose a backend and a local image
namespace:

```bash
REPO_OWNER=local VARIANT=cpu CUDA_VERSION=none \
  docker buildx bake -f docker-bake.hcl --load server full swap
```

For CUDA 12:

```bash
REPO_OWNER=local VARIANT=cu12 CUDA_VERSION=12.6.2 \
  docker buildx bake -f docker-bake.hcl --load server full swap
```

Published images use portable CPU/CUDA targets (`GGML_NATIVE=OFF`). Local builds
default to native CPU tuning unless overridden.
