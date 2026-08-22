#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly STUDIO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly INSTALL_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/ese-studio"
readonly BIN_DIR="$HOME/.local/bin"
readonly APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
readonly ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
readonly TARGET_BINARY="$INSTALL_ROOT/ese-studio"
readonly ESE_ROOT="$INSTALL_ROOT/ese"

if [[ "${1:-}" != "--skip-build" ]]; then
    "$STUDIO_DIR/install.sh"
fi

readonly BUILT_BINARY="$STUDIO_DIR/src-tauri/target/release/ese-studio"
if [[ ! -x "$BUILT_BINARY" ]]; then
    printf 'Release binary not found at %s\n' "$BUILT_BINARY" >&2
    printf 'Run without --skip-build to build it first.\n' >&2
    exit 1
fi

install -d "$INSTALL_ROOT" "$BIN_DIR" "$APP_DIR" "$ICON_DIR"
install -m 0755 "$BUILT_BINARY" "$TARGET_BINARY"
install -m 0644 "$STUDIO_DIR/app-icon.svg" "$ICON_DIR/ese-studio.svg"
if [[ ! -x "$STUDIO_DIR/runtime/ese/build/bin/llama-server" ]]; then
    printf 'Bundled ESE runtime was not staged at %s\n' "$STUDIO_DIR/runtime/ese" >&2
    exit 1
fi
install -d "$ESE_ROOT"
cp -a "$STUDIO_DIR/runtime/ese/." "$ESE_ROOT/"
chmod 0755 "$ESE_ROOT/ese" "$ESE_ROOT/build/bin/llama-server"

wrapper_tmp="$(mktemp)"
ese_wrapper_tmp="$(mktemp)"
desktop_tmp="$(mktemp)"
cleanup() { rm -f -- "$wrapper_tmp" "$ese_wrapper_tmp" "$desktop_tmp"; }
trap cleanup EXIT

printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'export WEBKIT_DISABLE_DMABUF_RENDERER=1' \
    'export ESE_BIN="${XDG_DATA_HOME:-$HOME/.local/share}/ese-studio/ese/ese"' \
    'exec "${XDG_DATA_HOME:-$HOME/.local/share}/ese-studio/ese-studio" "$@"' \
    >"$wrapper_tmp"
install -m 0755 "$wrapper_tmp" "$BIN_DIR/ese-studio"

printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'exec python3 "${XDG_DATA_HOME:-$HOME/.local/share}/ese-studio/ese/ese" "$@"' \
    >"$ese_wrapper_tmp"
install -m 0755 "$ese_wrapper_tmp" "$BIN_DIR/ese"

printf '%s\n' \
    '[Desktop Entry]' \
    'Type=Application' \
    'Version=1.0' \
    'Name=ESE Studio' \
    'GenericName=Local AI Model Manager' \
    'Comment=Discover, tune, and launch local GGUF models and CLI apps' \
    'Exec=ese-studio' \
    'Icon=ese-studio' \
    'Terminal=false' \
    'Categories=Development;' \
    'StartupNotify=true' \
    'StartupWMClass=ese-studio' \
    >"$desktop_tmp"
install -m 0644 "$desktop_tmp" "$APP_DIR/ese-studio.desktop"

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APP_DIR" >/dev/null 2>&1 || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -f -t "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor" >/dev/null 2>&1 || true

printf 'Installed ESE Studio to %s\n' "$TARGET_BINARY"
printf 'Installed ESE runtime to %s\n' "$ESE_ROOT"
printf 'ESE command: %s\n' "$BIN_DIR/ese"
printf 'Launcher: %s\n' "$APP_DIR/ese-studio.desktop"
