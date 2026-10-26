#!/bin/sh
# Download models for mynah. Usage: download-model.sh [parakeet|cohere|whisper|streaming|all]
set -eu

DATA="${XDG_DATA_HOME:-$HOME/.local/share}/mynah/models"
WHICH="${1:-parakeet}"

# fetch <hf repo> <file>: download a handy-computer gguf into $DATA.
fetch() {
    mkdir -p "$DATA"
    echo "==> $2"
    curl -fL --retry 3 -C - -o "$DATA/$2" \
        "https://huggingface.co/handy-computer/$1/resolve/main/$2"
    echo "Ready at $DATA/$2"
}

whisper() {
    OUT="${MYNAH_WHISPER_MODEL:-$DATA/ggml-large-v3-turbo-q5_0.bin}"
    mkdir -p "$(dirname "$OUT")"
    echo "==> ggml-large-v3-turbo-q5_0.bin"
    curl -fL --retry 3 -C - -o "$OUT" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
    echo "Whisper ready at $OUT"
}

parakeet() { fetch parakeet-ultra-gguf parakeet-ultra-Q8_0.gguf; }
cohere() { fetch cohere-transcribe-03-2026-gguf cohere-transcribe-03-2026-Q4_K_M.gguf; }
streaming() { fetch parakeet-unified-en-0.6b-gguf parakeet-unified-en-0.6b-Q8_0.gguf; }

case "$WHICH" in
    parakeet) parakeet ;;
    cohere) cohere ;;
    whisper) whisper ;;
    streaming) streaming ;;
    all) parakeet; cohere; whisper; streaming ;;
    *) echo "usage: $0 [parakeet|cohere|whisper|streaming|all]" >&2; exit 1 ;;
esac
