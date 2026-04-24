#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="/mnt/data/disk2/LibriTTS"

URLS=(
  "https://openslr.trmal.net/resources/60/train-clean-100.tar.gz"
  "https://openslr.trmal.net/resources/60/train-clean-360.tar.gz"
  "https://openslr.trmal.net/resources/60/train-other-500.tar.gz"
  "https://openslr.trmal.net/resources/60/test-clean.tar.gz"
  "https://openslr.trmal.net/resources/60/test-other.tar.gz"
  "https://openslr.trmal.net/resources/60/dev-clean.tar.gz"
  "https://openslr.trmal.net/resources/60/dev-other.tar.gz"
)

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

for url in "${URLS[@]}"; do
  file_name="$(basename "$url")"
  wget -c "$url" -O "$file_name"
done

for url in "${URLS[@]}"; do
  file_name="$(basename "$url")"
  tar -xzf "$file_name"
done

for url in "${URLS[@]}"; do
  file_name="$(basename "$url")"
  rm -f "$file_name"
done

echo "완료: $TARGET_DIR"
