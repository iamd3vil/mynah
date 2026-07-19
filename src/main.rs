mod asr;
mod audio;
mod icon;
mod overlay;
mod sock;
mod tray;
mod typer;

use std::sync::atomic::AtomicU32;
use std::sync::mpsc;
use std::sync::Arc;

use anyhow::{Context, Result};

/// Events that drive the main state machine.
pub enum Event {
    Toggle,
    Cancel,
    ModelReady,
    Transcribed(Result<String>),
    /// Streaming: newly committed text to type right now.
    StreamDelta(String),
    /// Streaming: session finished (after finalize or abort).
    StreamDone,
    Quit,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Phase {
    Loading,
    Idle,
    Recording,
    Transcribing,
}

/// Mic level (RMS, f32 bits) shared with the overlay for the waveform.
pub type Level = Arc<AtomicU32>;

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,zbus=warn,tracing=warn"),
    )
    .init();

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None | Some("daemon") => daemon(),
        Some(cmd @ ("toggle" | "cancel" | "status" | "quit")) => sock::send(cmd),
        // Offline test path: transcribe a 16 kHz mono wav and print the text.
        Some("transcribe") => {
            let path = args.next().context("usage: mynah transcribe <wav>")?;
            let text = asr::transcribe_file(std::path::Path::new(&path))?;
            println!("{text}");
            Ok(())
        }
        // Offline streaming test: feed a wav in chunks, print committed deltas.
        Some("stream-file") => {
            let path = args.next().context("usage: mynah stream-file <wav>")?;
            asr::stream_file(std::path::Path::new(&path))
        }
        // Print the configured model's capabilities (diagnostics).
        Some("caps") => asr::print_caps(),
        Some(other) => {
            eprintln!("usage: mynah [daemon|toggle|cancel|status|quit|transcribe <wav>|caps]");
            anyhow::bail!("unknown command: {other}");
        }
    }
}

fn daemon() -> Result<()> {
    let (tx, rx) = mpsc::channel::<Event>();

    let level: Level = Arc::new(AtomicU32::new(0));
    let phase = Arc::new(std::sync::Mutex::new(Phase::Loading));

    sock::listen(tx.clone(), phase.clone()).context("starting control socket")?;
    let tray = tray::spawn(tx.clone()).context("starting tray icon")?;

    // The model takes a few seconds to load; do it on the ASR worker so the
    // tray/socket are responsive immediately. Toggles before it's ready are
    // rejected with a notification.
    let asr = asr::Worker::spawn(tx.clone());
    let mut typer = typer::Typer::new().context("creating uinput keyboard")?;

    let stream_mode = asr::streaming_enabled();
    if stream_mode {
        log::info!("live streaming mode enabled");
    }

    let mut capture: Option<audio::Capture> = None;
    let mut overlay: Option<overlay::Overlay> = None;

    let set_phase = |p: Phase| {
        *phase.lock().unwrap() = p;
        tray.set_phase(p);
    };

    log::info!("mynah daemon started");

    loop {
        let ev = rx.recv().context("event channel closed")?;
        let current = *phase.lock().unwrap();
        match (current, ev) {
            (_, Event::Quit) => {
                log::info!("quitting");
                break;
            }
            (Phase::Loading, Event::ModelReady) => {
                set_phase(Phase::Idle);
                log::info!("model ready");
            }
            (Phase::Loading, Event::Toggle) => {
                notify("Model still loading, try again in a moment");
            }
            (Phase::Idle, Event::Toggle) => {
                let sink = if stream_mode {
                    asr.stream_start();
                    Some(asr.chunk_sink())
                } else {
                    None
                };
                match audio::Capture::start(level.clone(), sink) {
                    Ok(c) => {
                        capture = Some(c);
                        overlay = overlay::Overlay::spawn(level.clone())
                            .map_err(|e| log::error!("overlay failed: {e:#}"))
                            .ok();
                        set_phase(Phase::Recording);
                    }
                    Err(e) => {
                        log::error!("failed to start capture: {e:#}");
                        notify(&format!("Mic capture failed: {e}"));
                        if stream_mode {
                            asr.stream_abort();
                        }
                    }
                }
            }
            (Phase::Recording, Event::Toggle) => {
                if let Some(c) = capture.take() {
                    let samples = c.stop();
                    if let Some(o) = overlay.as_ref() {
                        o.set_transcribing();
                    }
                    set_phase(Phase::Transcribing);
                    if stream_mode {
                        asr.stream_end();
                    } else {
                        asr.transcribe(samples);
                    }
                }
            }
            (Phase::Recording, Event::Cancel) => {
                capture.take().map(|c| c.stop());
                if stream_mode {
                    asr.stream_abort();
                }
                overlay.take();
                set_phase(Phase::Idle);
            }
            // Live streaming: committed text can never be revised, type it now.
            (Phase::Recording | Phase::Transcribing, Event::StreamDelta(text)) => {
                if let Err(e) = typer.type_delta(&text) {
                    log::error!("typing failed: {e:#}");
                }
            }
            (Phase::Transcribing, Event::StreamDone) => {
                overlay.take();
                set_phase(Phase::Idle);
            }
            (Phase::Transcribing, Event::Transcribed(res)) => {
                overlay.take();
                set_phase(Phase::Idle);
                match res {
                    Ok(text) if !text.trim().is_empty() => {
                        log::info!("typing {} chars", text.len());
                        if let Err(e) = typer.type_text(text.trim()) {
                            log::error!("typing failed: {e:#}");
                            notify(&format!("Typing failed: {e}"));
                        }
                    }
                    Ok(_) => log::info!("empty transcription, nothing to type"),
                    Err(e) => {
                        log::error!("transcription failed: {e:#}");
                        notify(&format!("Transcription failed: {e}"));
                    }
                }
            }
            _ => {}
        }
    }

    capture.take().map(|c| c.stop());
    Ok(())
}

pub fn notify(body: &str) {
    let _ = std::process::Command::new("notify-send")
        .args(["-a", "mynah", "-i", "audio-input-microphone", "mynah", body])
        .spawn();
}
