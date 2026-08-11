#!/usr/bin/env bash
set -euo pipefail

# Stages the redistributable QAIRT/QNN runtime from a locally installed Qualcomm
# SDK into the Android project. This script does not download or redistribute
# Qualcomm binaries; it only copies files from the SDK the builder already has.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK="${QNN_SDK_ROOT:-${1:-}}"

if [[ -z "${SDK}" || ! -d "${SDK}" ]]; then
  echo "usage: QNN_SDK_ROOT=/path/to/qairt bash prepare-qnn-runtime.sh" >&2
  echo "   or: bash prepare-qnn-runtime.sh /path/to/qairt" >&2
  exit 2
fi

JNI_DST="${ROOT}/llama/src/main/jniLibs/arm64-v8a"
DSP_DST="${ROOT}/app/src/main/assets/qnn"
mkdir -p "${JNI_DST}" "${DSP_DST}"

android_count=0
while IFS= read -r -d '' file; do
  case "${file}" in
    *aarch64*|*arm64*|*android*)
      cp -f "${file}" "${JNI_DST}/"
      echo "host: ${file##*/}"
      android_count=$((android_count + 1))
      ;;
  esac
done < <(find "${SDK}" -type f -name 'libQnn*.so' -print0)

# Hexagon-side skeletons are not loaded by Android's linker. They are packaged
# as assets and extracted to app-private storage on first launch, then exposed
# to HTP through ADSP_LIBRARY_PATH.
dsp_count=0
while IFS= read -r -d '' file; do
  case "${file}" in
    *aarch64*|*arm64*|*android*) ;;
    *)
      cp -f "${file}" "${DSP_DST}/"
      echo "dsp:  ${file##*/}"
      dsp_count=$((dsp_count + 1))
      ;;
  esac
done < <(find "${SDK}" -type f \( -name '*Skel.so' -o -name 'libQnnHtpV*Stub.so' \) -print0)

if [[ ${android_count} -eq 0 ]]; then
  echo "warning: no arm64 Android libQnn*.so files found under ${SDK}" >&2
fi
if [[ ${dsp_count} -eq 0 ]]; then
  echo "warning: no Hexagon HTP skeleton files found under ${SDK}" >&2
fi

echo
echo "Staged QAIRT runtime:"
echo "  Android libraries: ${android_count} -> ${JNI_DST}"
echo "  DSP files:         ${dsp_count} -> ${DSP_DST}"
echo
echo "Build with:"
echo "  ./gradlew :app:assembleDebug -PexpertQnn=true -PqnnSdkRoot=${SDK}"
