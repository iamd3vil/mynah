# mynah

Voice-to-text daemon for KDE Plasma on Wayland. Press a hotkey, speak, press it
again — mynah transcribes locally with NVIDIA Parakeet TDT 0.6B v3 and types the
result into whatever has focus, terminals included.

Named after the mynah bird, which mimics human speech.

## How it works

- **Transcription**: [Parakeet TDT 0.6B v3](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
  (int8 ONNX) via [`transcribe-rs`](https://github.com/cjpais/transcribe-rs), fully offline.
- **Typing**: a uinput virtual keyboard — raw keystrokes, so it works in every
  app (no clipboard, no Ctrl+V). US layout; typographic characters are
  normalized to ASCII.
- **Hotkey**: a KDE global shortcut runs `mynah toggle`, which talks to the
  daemon over a Unix socket. No key-grabbing hacks.
- **UI**: a Plasma tray icon (StatusNotifierItem) plus a layer-shell pill at the
  bottom of the screen showing a live waveform while recording.

## Setup

```sh
# 1. Build and install the binary
cargo build --release
install -Dm755 target/release/mynah ~/.local/bin/mynah

# 2. Download the model (~700 MB, one time)
scripts/download-model.sh

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

| Command        | Effect                                        |
|----------------|-----------------------------------------------|
| `mynah toggle` | start recording / stop-transcribe-and-type    |
| `mynah cancel` | discard the current recording                 |
| `mynah status` | print `idle` / `recording` / `transcribing`   |
| `mynah quit`   | stop the daemon                               |

Left-clicking the tray icon also toggles.

## Configuration

Environment variables (set in the systemd unit if needed):

- `MYNAH_MODEL_DIR` — model directory (default
  `~/.local/share/mynah/models/parakeet-tdt-0.6b-v3-int8`)
- `RUST_LOG` — log level (default `info`); `journalctl --user -u mynah` to read.
