<p align="center">
  <img src="assets/logo.svg" width="120" alt="mynah — a plump white bird with an amber beak">
</p>

# mynah

Voice-to-text daemon for KDE Plasma on Wayland. Press a hotkey, speak, press it
again — mynah transcribes locally (Parakeet Ultra or Cohere Transcribe,
GPU-accelerated via Vulkan) and types the result into whatever has focus,
terminals included.

Named after the mynah bird, which mimics human speech.

## How it works

- **Transcription**: [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
  (unified ggml inference engine, Vulkan GPU offload) via the
  [`transcribe-cpp`](https://crates.io/crates/transcribe-cpp) crate, fully
  offline. Engines: Parakeet Ultra (fast), Cohere Transcribe (most accurate)
  or Whisper large-v3-turbo; optional live streaming with
  parakeet-unified-en.
- **Typing**: a uinput virtual keyboard — raw keystrokes, so it works in every
  app (no clipboard, no Ctrl+V). US layout; typographic characters are
  normalized to ASCII.
- **Hotkey**: a KDE global shortcut runs `mynah toggle`, which talks to the
  daemon over a Unix socket. No key-grabbing hacks.
- **UI**: a Plasma tray icon (StatusNotifierItem) plus a layer-shell pill at the
  bottom of the screen showing a live waveform while recording.

## Setup

```sh
# 0. Build prerequisites: a C++ toolchain, cmake, and for the (default)
#    Vulkan backend: vulkan-headers, spirv-headers, shaderc (glslc).

# 1. Build and install the binary
cargo build --release
install -Dm755 target/release/mynah ~/.local/bin/mynah

# 2. Download a model (one time): parakeet (default, ~740 MB),
#    cohere (~1.6 GB), whisper (~550 MB), streaming (~730 MB), or all
scripts/download-model.sh parakeet
scripts/download-model.sh cohere    # optional, for MYNAH_ENGINE=cohere

# 3. uinput access (skip if /dev/uinput is already writable)
sudo tee /etc/udev/rules.d/60-mynah-uinput.rules <<'EOF'
KERNEL=="uinput", GROUP="input", MODE="0660", TAG+="uaccess"
EOF
sudo udevadm control --reload && sudo udevadm trigger /dev/uinput

# 4. Run as a user service
install -Dm644 dist/mynah.service ~/.config/systemd/user/mynah.service
systemctl --user daemon-reload
systemctl --user enable --now mynah

# 5. Bind a key in KDE
# System Settings → Keyboard → Shortcuts → Add New → Command or Script:
#   command: ~/.local/bin/mynah toggle
# and assign e.g. Meta+H.
#
# Or headless, via kglobalaccel's D-Bus API (Meta+H = 0x10000048):
#   busctl --user call org.kde.kglobalaccel /kglobalaccel org.kde.KGlobalAccel \
#     doRegister as 4 "mynah-toggle.desktop" "_launch" "Mynah Toggle" "Launch"
#   busctl --user call org.kde.kglobalaccel /kglobalaccel org.kde.KGlobalAccel \
#     setShortcut asaiu 4 "mynah-toggle.desktop" "_launch" "Mynah Toggle" "Launch" 1 268435528 6
# Notes for Plasma 6 (learned the hard way):
# - kglobalacceld may run inside kwin_wayland; editing kglobalshortcutsrc and
#   restarting plasma-kglobalaccel.service does nothing until next login.
# - flags must include SetPresent (2) or the key is registered but never
#   grabbed: 6 = SetPresent | NoAutoloading.
# - the mynah-toggle.desktop launcher must exist in
#   ~/.local/share/applications (and/or ~/.local/share/kglobalaccel).
```

## Usage

| Command                  | Effect                                        |
|--------------------------|-----------------------------------------------|
| `mynah toggle`           | start recording / stop-transcribe-and-type    |
| `mynah cancel`           | discard the current recording                 |
| `mynah status`           | print `idle` / `recording` / `transcribing`   |
| `mynah quit`             | stop the daemon                               |
| `mynah transcribe <wav>...` | transcribe 16 kHz mono wavs, one line each (testing) |
| `mynah stream-file <wav>`| offline streaming test with feed timings      |
| `mynah caps`             | print the configured model's capabilities     |

Left-clicking the tray icon also toggles.

## Configuration

Environment variables (set in the systemd unit if needed):

- `MYNAH_ENGINE` — `parakeet` (default; Parakeet Ultra), `cohere` (Cohere
  Transcribe 2B, most accurate, ~3x slower), or `whisper` (large-v3-turbo;
  slower and less accurate than cohere, but biased by `vocabulary.txt`). Each
  needs its model downloaded first (`scripts/download-model.sh`). See
  [Benchmarks](#benchmarks).
- `MYNAH_MODEL` — override the model path entirely (any gguf transcribe.cpp
  supports; whisper.cpp `.bin` files also load).
- `MYNAH_STREAM=1` — live streaming: text is typed *while you speak* (stable
  committed prefix only — never revised, so no stray backspaces). Uses the
  parakeet-unified-en model (`scripts/download-model.sh streaming`).
  Trade-off: streaming is less accurate than batch, and punctuation is
  sparser. Test throughput with `mynah stream-file <wav>`.
- `MYNAH_LANG` — transcription language hint (default `en`; matched to the
  model's language list, so locale models like nemotron get `en-US`)
- `MYNAH_SOUNDS` — set to `0`, `false`, or `off` to disable the brief start and
  stop recording tones (enabled by default)
- `RUST_LOG` — log level (default `info`); `journalctl --user -u mynah` to read.

### Vocabulary corrections

Create `~/.config/mynah/vocabulary.txt` with one `misheard => desired` rule per
line. Blank lines and lines beginning with `#` are ignored:

```text
# ASR output => spelling to type
kite connect => Kite Connect
cargo toml => Cargo.toml
mynah => mynah
```

Rules match complete ASCII case-insensitive words or phrases, with longer
phrases taking precedence. Corrections are applied before batch transcription
is typed. Models that accept a vocabulary bias (whisper, qwen3-asr) also get
the desired terms as decoding hints. Streaming transcription types an
irreversible committed prefix, so it does not apply vocabulary corrections.
Restart the service after changing the file.

To switch engines: `systemctl --user edit mynah`, add
`[Service]` / `Environment=MYNAH_ENGINE=cohere`, then
`systemctl --user restart mynah`.

All engines run on the GPU via Vulkan when available (`Backend::Auto` probes
discrete GPUs, then integrated, then CPU). CPU-only whisper is ~20s per
utterance and not dictation-friendly.

## Benchmarks

`scripts/bench.py fetch` pulls 200 dictation-length clips (3–15s) from the
Open ASR Leaderboard test sets; `scripts/bench.py run MODEL...` reports WER
and speed through `mynah transcribe`. Radeon 860M iGPU, Vulkan, Oct 2026.
WER ignores case, punctuation and fillers but not number formatting, so it
runs higher than leaderboard figures. RTFx = audio seconds per second.

| model                           | voxpopuli | ami   | earnings22 | libri-other | all   | RTFx |
|---------------------------------|-----------|-------|------------|-------------|-------|------|
| parakeet-tdt-0.6b-v3 Q8 (old)   | 5.81      | 9.86  | 17.35      | 3.39        | 8.53  | 55   |
| **parakeet-ultra** Q8           | 6.26      | 8.57  | 11.92      | 3.49        | 7.19  | 54   |
| parakeet-unified-en Q8          | 5.47      | 8.71  | 13.64      | 3.49        | 7.40  | 57   |
| granite-speech-5.0-470m Q8 ¹    | 7.15      | 20.71 | 13.51      | 5.19        | 10.83 | 57   |
| whisper-large-v3-turbo q5       | 9.83      | 13.29 | 12.45      | 3.49        | 9.25  | 5    |
| **cohere-transcribe** Q4_K_M    | 5.92      | 8.00  | 10.07      | 2.30        | 6.21  | 20   |
| granite-speech-4.1-2b-nar Q4 ¹  | 4.58      | 6.86  | 9.67       | 3.09        | 5.76  | 11   |
| granite-speech-4.1-2b Q4 ¹      | 5.92      | 7.57  | 9.54       | 3.39        | 6.32  | 5    |
| qwen3-asr-0.6b Q8               | 7.82      | 10.00 | 12.58      | 5.29        | 8.59  | 12   |
| qwen3-asr-1.7b Q4_K_M           | 7.49      | 9.57  | 11.13      | 4.59        | 7.88  | 4    |

¹ Mostly lowercase, little or no punctuation: unsuitable for typing.

Streaming (`mynah stream-file`, 25 clips): parakeet-unified-en 9.05% WER,
p95 feed 140ms; nemotron-3.5-asr-streaming 13.32%, p95 87ms (160ms budget).
