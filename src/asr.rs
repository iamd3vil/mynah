//! Transcription worker. Owns the speech model on a dedicated thread so
//! inference never blocks the main state machine.
//!
//! Two engines are supported, chosen with MYNAH_ENGINE:
//! - "parakeet" (default): Parakeet TDT 0.6B v3, near-instant on CPU.
//! - "whisper": whisper.cpp (e.g. large-v3-turbo), slower but markedly more
//!   robust on accented English.

use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};

use anyhow::{Context, Result};
use transcribe_rs::onnx::parakeet::ParakeetModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::{SpeechModel, TranscribeOptions};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

use crate::Event;

fn data_dir() -> PathBuf {
    let data = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME not set")).join(".local/share")
        });
    data.join("mynah/models")
}

pub fn parakeet_dir() -> PathBuf {
    std::env::var("MYNAH_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir().join("parakeet-tdt-0.6b-v3-int8"))
}

pub fn whisper_path() -> PathBuf {
    std::env::var("MYNAH_WHISPER_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir().join("ggml-large-v3-turbo-q5_0.bin"))
}

/// whisper.cpp scales to physical cores; hyperthreads only add spin overhead.
/// Overridable with MYNAH_WHISPER_THREADS.
fn whisper_threads() -> i32 {
    if let Ok(n) = std::env::var("MYNAH_WHISPER_THREADS") {
        if let Ok(n) = n.parse() {
            return n;
        }
    }
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    (logical / 2).max(4) as i32
}

fn language() -> String {
    // Pinned language (default "en") — auto-detect adds latency and can
    // misfire on short dictation clips. Parakeet ignores this.
    std::env::var("MYNAH_LANG").unwrap_or_else(|_| "en".into())
}

/// Engine wrapper. Whisper is driven through whisper-rs directly so we
/// control thread count and sampling — transcribe-rs hardcodes beam search
/// (beam_size 3), which is ~3x slower than greedy for a small accuracy gain.
enum Engine {
    Parakeet(ParakeetModel),
    Whisper(Whisper),
}

impl Engine {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        match self {
            Engine::Parakeet(m) => {
                let opts = TranscribeOptions {
                    language: Some(language()),
                    ..Default::default()
                };
                Ok(m.transcribe(samples, &opts)?.text)
            }
            Engine::Whisper(m) => m.transcribe(samples),
        }
    }
}

struct Whisper {
    state: WhisperState,
    // Owns the C memory backing `state`; must outlive it.
    _context: WhisperContext,
}

impl Whisper {
    fn load(path: &std::path::Path) -> Result<Self> {
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu = true; // no-op unless built with a GPU backend
        ctx_params.flash_attn = true;
        let context = WhisperContext::new_with_params(
            path.to_str().context("non-utf8 model path")?,
            ctx_params,
        )
        .map_err(|e| anyhow::anyhow!("loading whisper model: {e}"))?;
        let state = context
            .create_state()
            .map_err(|e| anyhow::anyhow!("creating whisper state: {e}"))?;
        Ok(Self { state, _context: context })
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        // MYNAH_WHISPER_BEAM=N opts into beam search for max accuracy at ~Nx
        // the decode cost; greedy is the right default for dictation latency.
        let beam: usize = std::env::var("MYNAH_WHISPER_BEAM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let strategy = if beam > 1 {
            SamplingStrategy::BeamSearch { beam_size: beam as i32, patience: -1.0 }
        } else {
            SamplingStrategy::Greedy { best_of: 1 }
        };

        let lang = language();
        let mut params = FullParams::new(strategy);
        params.set_language(Some(&lang));
        params.set_n_threads(whisper_threads());
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);

        self.state
            .full(params, samples)
            .map_err(|e| anyhow::anyhow!("whisper inference: {e}"))?;

        let mut text = String::new();
        for i in 0..self.state.full_n_segments() {
            let segment = self
                .state
                .get_segment(i)
                .context("whisper segment out of bounds")?;
            text.push_str(segment.to_str().map_err(|e| anyhow::anyhow!("segment text: {e}"))?);
        }
        Ok(text.trim().to_string())
    }
}

fn load() -> Result<Engine> {
    let engine = std::env::var("MYNAH_ENGINE").unwrap_or_else(|_| "parakeet".into());
    match engine.as_str() {
        "parakeet" => {
            let dir = parakeet_dir();
            log::info!("loading parakeet from {}", dir.display());
            anyhow::ensure!(
                dir.join("vocab.txt").exists(),
                "parakeet model missing at {} (run scripts/download-model.sh parakeet)",
                dir.display()
            );
            let model = ParakeetModel::load(&dir, &Quantization::Int8)
                .context("loading parakeet model")?;
            Ok(Engine::Parakeet(model))
        }
        "whisper" => {
            let path = whisper_path();
            log::info!("loading whisper from {}", path.display());
            anyhow::ensure!(
                path.exists(),
                "whisper model missing at {} (run scripts/download-model.sh whisper)",
                path.display()
            );
            Ok(Engine::Whisper(Whisper::load(&path)?))
        }
        other => anyhow::bail!("unknown MYNAH_ENGINE {other:?} (use parakeet or whisper)"),
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
                let mut model = match load() {
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
    let samples = transcribe_rs::audio::read_wav_samples(path)?;
    let t0 = std::time::Instant::now();
    let mut engine = load()?;
    let loaded = t0.elapsed();
    let t1 = std::time::Instant::now();
    let text = engine.transcribe(&samples)?;
    eprintln!(
        "load: {loaded:.1?}, inference: {:.1?} for {:.1}s audio",
        t1.elapsed(),
        samples.len() as f32 / 16000.0
    );
    Ok(text)
}
