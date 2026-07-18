#!/bin/sh
# Download the Parakeet TDT 0.6B v3 int8 ONNX model that mynah uses.
set -eu

BASE="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main"
DIR="${MYNAH_MODEL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/mynah/models/parakeet-tdt-0.6b-v3-int8}"

mkdir -p "$DIR"
for f in encoder-model.int8.onnx decoder_joint-model.int8.onnx nemo128.onnx vocab.txt; do
    echo "==> $f"
    curl -fL --retry 3 -C - -o "$DIR/$f" "$BASE/$f"
done

echo "Model ready in $DIR"
