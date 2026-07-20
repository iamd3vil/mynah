<p align="center">
  <img src="assets/logo.svg" width="120" alt="mynah — a plump white bird with an amber beak">
</p>

# mynah

Voice-to-text daemon for KDE Plasma on Wayland. Press a hotkey, speak, press it
again — mynah transcribes locally (Parakeet TDT v3 or Whisper large-v3-turbo,
GPU-accelerated via Vulkan) and types the result into whatever has focus,
terminals included.

Named after the mynah bird, which mimics human speech.

## How it works

- **Transcription**: [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
  (unified ggml inference engine, Vulkan GPU offload) via the
  [`transcribe-cpp`](https://crates.io/crates/transcribe-cpp) crate, fully
  offline. Engines: Parakeet TDT 0.6B v3 (fast) or Whisper large-v3-turbo
  (better on accented English); optional live streaming with
  nemotron-3.5-asr-streaming.
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
#    transcribe-cpp-sys 0.1.3 also forgets to link CBLAS on Linux;
#    build.rs works around it (needs a system cblas).

# 1. Build and install the binary
cargo build --release
install -Dm755 target/release/mynah ~/.local/bin/mynah

# 2. Download a model (one time): parakeet (default, ~640 MB),
#    whisper (~550 MB), or all
scripts/download-model.sh parakeet
scripts/download-model.sh whisper   # optional, for MYNAH_ENGINE=whisper

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
| `mynah transcribe <wav>` | transcribe a 16 kHz mono wav (testing)        |
| `mynah stream-file <wav>`| offline streaming test with feed timings      |
| `mynah caps`             | print the configured model's capabilities     |

Left-clicking the tray icon also toggles.

## Configuration

Environment variables (set in the systemd unit if needed):

- `MYNAH_ENGINE` — `parakeet` (default; ~0.5s per utterance on an iGPU) or
  `whisper` (large-v3-turbo, ~2s; much better on accented English). Each needs
  its model downloaded first (`scripts/download-model.sh`).
- `MYNAH_MODEL` — override the model path entirely (any gguf transcribe.cpp
  supports; whisper.cpp `.bin` files also load).
- `MYNAH_STREAM=1` — live streaming: text is typed *while you speak* (stable
  committed prefix only — never revised, so no stray backspaces). Uses the
  nemotron-3.5-asr-streaming model (`scripts/download-model.sh streaming`).
  Trade-off: streaming models are less accurate than batch whisper, and
  punctuation is sparser. Test throughput with `mynah stream-file <wav>`.
- `MYNAH_LANG` — transcription language hint (default `en`; `en-US` locale
  form auto-selected for streaming)
- `RUST_LOG` — log level (default `info`); `journalctl --user -u mynah` to read.

To switch engines: `systemctl --user edit mynah`, add
`[Service]` / `Environment=MYNAH_ENGINE=whisper`, then
`systemctl --user restart mynah`.

All engines run on the GPU via Vulkan when available (`Backend::Auto` probes
discrete GPUs, then integrated, then CPU). Timings above are from a Radeon
860M iGPU; CPU-only whisper is ~20s per utterance and not dictation-friendly.
