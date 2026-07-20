#!/bin/sh
# Download models for mynah. Usage: download-model.sh [parakeet|whisper|all]
set -eu

DATA="${XDG_DATA_HOME:-$HOME/.local/share}/mynah/models"
WHICH="${1:-parakeet}"

parakeet() {
    OUT="$DATA/parakeet-tdt-0.6b-v3-Q8_0.gguf"
    mkdir -p "$DATA"
    echo "==> parakeet-tdt-0.6b-v3-Q8_0.gguf"
    curl -fL --retry 3 -C - -o "$OUT" \
        "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf"
    echo "Parakeet ready at $OUT"
}

whisper() {
    OUT="${MYNAH_WHISPER_MODEL:-$DATA/ggml-large-v3-turbo-q5_0.bin}"
    mkdir -p "$(dirname "$OUT")"
    echo "==> ggml-large-v3-turbo-q5_0.bin"
    curl -fL --retry 3 -C - -o "$OUT" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
    echo "Whisper ready at $OUT"
}

streaming() {
    OUT="$DATA/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"
    mkdir -p "$DATA"
    echo "==> nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"
    curl -fL --retry 3 -C - -o "$OUT" \
        "https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"
    echo "Streaming model ready at $OUT"
}

case "$WHICH" in
    parakeet) parakeet ;;
    whisper) whisper ;;
    streaming) streaming ;;
    all) parakeet; whisper; streaming ;;
    *) echo "usage: $0 [parakeet|whisper|streaming|all]" >&2; exit 1 ;;
esac
