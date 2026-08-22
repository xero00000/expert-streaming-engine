#!/usr/bin/env bash
set -euo pipefail

readonly APP_NAME="ESE Studio"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly CHECK_ONLY="${1:-}"

if [[ ! -r /etc/os-release ]]; then
    printf 'Unable to identify this Linux distribution (/etc/os-release is missing).\n' >&2
    exit 1
fi

# shellcheck source=/dev/null
source /etc/os-release

missing_commands=()
missing_libraries=()
optional_commands=()

for command_name in node cargo rustc pkg-config cmake python3; do
    command -v "$command_name" >/dev/null 2>&1 || missing_commands+=("$command_name")
done
pnpm_missing=0
command -v pnpm >/dev/null 2>&1 || pnpm_missing=1

for library_name in webkit2gtk-4.1 gtk+-3.0 openssl; do
    pkg-config --exists "$library_name" 2>/dev/null || missing_libraries+=("$library_name")
done

for command_name in nvidia-smi; do
    command -v "$command_name" >/dev/null 2>&1 || optional_commands+=("$command_name")
done

printf '%s dependency preflight\n' "$APP_NAME"
printf '  Distribution: %s\n' "${PRETTY_NAME:-${ID:-unknown}}"

if ((${#missing_commands[@]} == 0 && ${#missing_libraries[@]} == 0)); then
    printf '  Required build dependencies: ready\n'
else
    ((${#missing_commands[@]})) && printf '  Missing commands: %s\n' "${missing_commands[*]}"
    ((${#missing_libraries[@]})) && printf '  Missing libraries: %s\n' "${missing_libraries[*]}"
fi
((pnpm_missing)) && printf '  Missing JavaScript package manager: pnpm\n'

if ((${#optional_commands[@]})); then
    printf '  Optional runtime tools not found: %s\n' "${optional_commands[*]}"
    printf '  Studio can build without them; model serving or NVIDIA acceleration may be unavailable.\n'
fi

if [[ "$CHECK_ONLY" == "--check" ]]; then
    if ((${#missing_commands[@]} || ${#missing_libraries[@]} || pnpm_missing)); then
        exit 2
    fi
    exit 0
fi

install_command=()
case "${ID:-}" in
    ubuntu|debian|linuxmint|pop)
        install_command=(sudo apt-get install -y build-essential cmake python3 curl wget file pkg-config libwebkit2gtk-4.1-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev nodejs npm cargo rustc patchelf)
        ;;
    fedora|nobara|rhel|centos|rocky|almalinux)
        install_command=(sudo dnf install -y gcc gcc-c++ make cmake python3 curl wget file pkgconf-pkg-config webkit2gtk4.1-devel openssl-devel libappindicator-gtk3-devel librsvg2-devel nodejs npm cargo rust patchelf)
        ;;
    arch|manjaro|endeavouros)
        install_command=(sudo pacman -S --needed base-devel cmake python curl wget file pkgconf webkit2gtk-4.1 openssl libappindicator-gtk3 librsvg nodejs npm rust patchelf)
        ;;
    opensuse*|sles)
        install_command=(sudo zypper install -y gcc gcc-c++ make cmake python3 curl wget file pkg-config webkit2gtk3-devel libopenssl-devel libappindicator3-1 librsvg-devel nodejs npm rust cargo patchelf)
        ;;
esac

if ((${#missing_commands[@]} || ${#missing_libraries[@]})); then
    if ((${#install_command[@]} == 0)); then
        printf 'Automatic dependency installation is not supported for %s.\n' "${ID:-this distribution}" >&2
        printf 'See %s/README.md for the required packages.\n' "$SCRIPT_DIR" >&2
        exit 1
    fi
    if [[ ! -t 0 ]]; then
        printf 'Dependencies are missing and input is not interactive; refusing to install automatically.\n' >&2
        exit 2
    fi
    printf '\nInstall the missing build dependencies now? [y/N] '
    read -r reply
    if [[ ! "$reply" =~ ^[Yy]$ ]]; then
        printf 'No packages were installed.\n'
        exit 2
    fi
    "${install_command[@]}"
fi

if ! command -v pnpm >/dev/null 2>&1; then
    if [[ ! -t 0 ]]; then
        printf 'pnpm is missing and input is not interactive; refusing to install automatically.\n' >&2
        exit 2
    fi
    printf '\nInstall pnpm for the current user now? [y/N] '
    read -r reply
    if [[ ! "$reply" =~ ^[Yy]$ ]]; then
        printf 'pnpm was not installed.\n'
        exit 2
    fi
    if command -v corepack >/dev/null 2>&1; then
        corepack enable pnpm
    elif command -v npm >/dev/null 2>&1; then
        npm install --global --prefix "$HOME/.local" pnpm
    else
        printf 'Neither corepack nor npm is available. Install pnpm, then rerun this script.\n' >&2
        exit 1
    fi
fi

printf '\nBuilding the ESE runtime…\n'
python3 "$SCRIPT_DIR/../ese" build --backend auto --build-dir "$SCRIPT_DIR/../build-package"

printf '\nBuilding %s…\n' "$APP_NAME"
cd "$SCRIPT_DIR"
pnpm install --frozen-lockfile
pnpm tauri build --config src-tauri/tauri.unsigned.conf.json
printf '\nBuild complete. DEB and RPM packages are under %s/src-tauri/target/release/bundle/.\n' "$SCRIPT_DIR"
