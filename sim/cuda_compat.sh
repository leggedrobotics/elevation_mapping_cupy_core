#!/usr/bin/env bash
# `pixi run cuda-compat`: on JetPack 6 (Orin, kernel driver CUDA 12.6), unpack
# NVIDIA's CUDA forward-compatibility user-mode driver into .pixi/cuda-compat/
# without root; the activation puts it ahead of the system's libcuda. JetPack 7
# (Thor) and desktops run the environment's CUDA 13 as they are.
# The package's CUDA minor must be >= the environment's cuda-version (pixi.toml).
set -euo pipefail

DEB=cuda-compat-orin-13-5_13.5.47659993-1_arm64.deb
dest="$PIXI_PROJECT_ROOT/.pixi/cuda-compat"

if ! grep -qs '^# R36 ' /etc/nv_tegra_release; then
  echo "not JetPack 6: no compat libraries needed"
  exit 0
fi
if [ "$(cat "$dest/.deb" 2>/dev/null)" = "$DEB" ]; then
  echo "$DEB is already in $dest"
  exit 0
fi
tmp="$(mktemp -d "$PIXI_PROJECT_ROOT/.pixi/cuda-compat.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/$DEB" "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/$DEB"
dpkg-deb -x "$tmp/$DEB" "$tmp/root"
ln -s "$(cd "$tmp/root" && ls -d usr/local/cuda-*/compat_orin)" "$tmp/root/lib"
echo "$DEB" > "$tmp/root/.deb"
rm -rf "$dest"
mv "$tmp/root" "$dest"
echo "CUDA compat libraries in $dest/lib"
