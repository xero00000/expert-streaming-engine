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

# Clear only files this helper owns so changing QAIRT versions cannot leave a
# stale library beside the new runtime.
rm -f "${JNI_DST}"/libQnn*.so "${DSP_DST}"/*.so

android_count=0
while IFS= read -r -d '' file; do
  lower="${file,,}"
  # QAIRT SDK layouts have changed names over time. Require an ARM64 marker and
  # reject obvious x86 trees; matching merely "android" would incorrectly copy
  # x86_64 redistributables into an arm64 APK.
  if [[ "${lower}" == *aarch64* || "${lower}" == *arm64* || "${lower}" == *armv8* ]]; then
    if [[ "${lower}" != *x86* && "${lower}" != *x64* ]]; then
      cp -f "${file}" "${JNI_DST}/"
      echo "host: ${file##*/}"
      android_count=$((android_count + 1))
    fi
  fi
done < <(find "${SDK}" -type f -name 'libQnn*.so' -print0)

# Hexagon-side skeletons are not loaded by Android's linker. They are packaged
# as assets and extracted to app-private storage on first launch, then exposed
# to HTP through ADSP_LIBRARY_PATH.
dsp_count=0
while IFS= read -r -d '' file; do
  lower="${file,,}"
  # True DSP/Hexagon skeletons normally live under hexagon-v* directories and
  # should not be confused with Android ARM64 stub libraries.
  if [[ "${lower}" == *hexagon* || "${lower}" == *dsp* ]]; then
    cp -f "${file}" "${DSP_DST}/"
    echo "dsp:  ${file##*/}"
    dsp_count=$((dsp_count + 1))
  fi
done < <(find "${SDK}" -type f -name '*Skel.so' -print0)

if [[ ${android_count} -eq 0 ]]; then
  echo "warning: no arm64 libQnn*.so files found under ${SDK}" >&2
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
