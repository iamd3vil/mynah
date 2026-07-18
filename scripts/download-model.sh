#!/bin/sh
# Download models for mynah. Usage: download-model.sh [parakeet|whisper|all]
set -eu

DATA="${XDG_DATA_HOME:-$HOME/.local/share}/mynah/models"
WHICH="${1:-parakeet}"

parakeet() {
    BASE="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main"
    DIR="${MYNAH_MODEL_DIR:-$DATA/parakeet-tdt-0.6b-v3-int8}"
    mkdir -p "$DIR"
    for f in encoder-model.int8.onnx decoder_joint-model.int8.onnx nemo128.onnx vocab.txt; do
        echo "==> $f"
        curl -fL --retry 3 -C - -o "$DIR/$f" "$BASE/$f"
    done
    echo "Parakeet ready in $DIR"
}

whisper() {
    OUT="${MYNAH_WHISPER_MODEL:-$DATA/ggml-large-v3-turbo-q5_0.bin}"
    mkdir -p "$(dirname "$OUT")"
    echo "==> ggml-large-v3-turbo-q5_0.bin"
    curl -fL --retry 3 -C - -o "$OUT" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
    echo "Whisper ready at $OUT"
}

case "$WHICH" in
    parakeet) parakeet ;;
    whisper) whisper ;;
    all) parakeet; whisper ;;
    *) echo "usage: $0 [parakeet|whisper|all]" >&2; exit 1 ;;
esac
