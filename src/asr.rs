//! Transcription worker. Owns the speech model on a dedicated thread so
//! inference never blocks the main state machine.
//!
//! Inference runs on transcribe.cpp (unified ggml engine) — one API for every
//! model family, GPU offload via the `vulkan` feature. MYNAH_ENGINE picks the
//! default model:
//! - "parakeet" (default): Parakeet TDT 0.6B v3, fast.
//! - "whisper": Whisper large-v3-turbo, better on accented English.
//! MYNAH_MODEL overrides the model path entirely (any supported gguf).

use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};

use anyhow::{Context, Result};
use transcribe_cpp::{Model, RunOptions, Session};

use crate::Event;

fn data_dir() -> PathBuf {
    let data = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME not set")).join(".local/share")
        });
    data.join("mynah/models")
}

fn model_path() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("MYNAH_MODEL") {
        return Ok(PathBuf::from(path));
    }
    let engine = std::env::var("MYNAH_ENGINE").unwrap_or_else(|_| "parakeet".into());
    match engine.as_str() {
        "parakeet" => Ok(data_dir().join("parakeet-tdt-0.6b-v3-Q8_0.gguf")),
        // whisper.cpp .bin files load directly (backward compatible).
        "whisper" => Ok(data_dir().join("ggml-large-v3-turbo-q5_0.bin")),
        other => anyhow::bail!("unknown MYNAH_ENGINE {other:?} (use parakeet or whisper)"),
    }
}

fn language() -> String {
    std::env::var("MYNAH_LANG").unwrap_or_else(|_| "en".into())
}

struct Asr {
    session: Session,
}

impl Asr {
    fn load() -> Result<Self> {
        let path = model_path()?;
        log::info!("loading model {}", path.display());
        anyhow::ensure!(
            path.exists(),
            "model missing at {} (run scripts/download-model.sh)",
            path.display()
        );
        let model = Model::load(&path)
            .with_context(|| format!("loading model {}", path.display()))?;
        let session = model.session().context("opening session")?;
        Ok(Self { session })
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let options = RunOptions {
            // Pinned language (default "en") — auto-detect adds latency and
            // can misfire on short dictation clips.
            language: Some(language()),
            ..Default::default()
        };
        let transcript = self
            .session
            .run(samples, &options)
            .map_err(|e| anyhow::anyhow!("inference: {e}"))?;
        Ok(transcript.text.trim().to_string())
    }
}

pub struct Worker {
    jobs: Sender<Vec<f32>>,
}

impl Worker {
    pub fn spawn(events: Sender<Event>) -> Self {
        let (jobs, job_rx) = mpsc::channel::<Vec<f32>>();

        std::thread::Builder::new()
            .name("asr".into())
            .spawn(move || {
                let mut model = match Asr::load() {
                    Ok(m) => {
                        events.send(Event::ModelReady).ok();
                        m
                    }
                    Err(e) => {
                        log::error!("model load failed: {e:#}");
                        crate::notify(&format!("Model load failed: {e}"));
                        return;
                    }
                };

                while let Ok(samples) = job_rx.recv() {
                    let started = std::time::Instant::now();
                    let result = model.transcribe(&samples);
                    if let Ok(text) = &result {
                        log::info!(
                            "transcribed {:.1}s audio in {:?}: {text:?}",
                            samples.len() as f32 / 16000.0,
                            started.elapsed()
                        );
                    }
                    events.send(Event::Transcribed(result)).ok();
                }
            })
            .expect("spawning asr thread");

        Worker { jobs }
    }

    pub fn transcribe(&self, samples: Vec<f32>) {
        self.jobs.send(samples).expect("asr thread died");
    }
}

/// One-shot file transcription for `mynah transcribe` (testing/debugging).
pub fn transcribe_file(path: &std::path::Path) -> Result<String> {
    let samples = read_wav_16k_mono(path)?;
    let t0 = std::time::Instant::now();
    let mut asr = Asr::load()?;
    let loaded = t0.elapsed();
    let t1 = std::time::Instant::now();
    let text = asr.transcribe(&samples)?;
    eprintln!(
        "load: {loaded:.1?}, inference: {:.1?} for {:.1}s audio",
        t1.elapsed(),
        samples.len() as f32 / 16000.0
    );
    Ok(text)
}

/// Minimal RIFF/WAVE reader: 16 kHz mono 16-bit PCM only (test tooling).
fn read_wav_16k_mono(path: &std::path::Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(bytes.len() > 44 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE", "not a wav file");
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        match id {
            b"fmt " => {
                let channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
                let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
                anyhow::ensure!(
                    (channels, rate, bits) == (1, 16000, 16),
                    "expected 16kHz mono s16 wav, got {channels}ch {rate}Hz {bits}bit"
                );
            }
            b"data" => {
                return Ok(body
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                    .collect());
            }
            _ => {}
        }
        pos += 8 + len + (len & 1);
    }
    anyhow::bail!("no data chunk in wav")
}
