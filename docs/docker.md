# ESE containers

The canonical container instructions live in [docker/README.md](../docker/README.md).

ESE's container workflow builds this repository and publishes CPU, CUDA 12,
and CUDA 13 variants to the `xero00000/ik-llama-cpp` package on GitHub Container
Registry. The package name retains the inherited native-engine name; the image
contents come from Expert Streaming Engine.

The old inherited `ggerganov/llama.cpp` image instructions were removed because
those images do not contain ESE's launcher or native resource-controller work.
