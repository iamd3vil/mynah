//! Transcription worker. Owns the speech model on a dedicated thread so
//! inference never blocks the main state machine.
//!
//! Inference runs on transcribe.cpp (unified ggml engine) — one API for every
//! model family, GPU offload via the `vulkan` feature. MYNAH_ENGINE picks the
//! default model:
//! - "parakeet" (default): Parakeet Ultra (post-trained TDT 0.6B v3), fast.
//! - "cohere": Cohere Transcribe 2B, most accurate (~3x parakeet's latency).
//! - "whisper": Whisper large-v3-turbo; slower and less accurate than cohere,
//!   but takes vocabulary.txt terms as a decoding bias.
//!
//! Accuracy/speed numbers: scripts/bench.py. MYNAH_MODEL overrides the model
//! path entirely (any supported gguf).

use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};

use anyhow::{Context, Result};
use transcribe_cpp::{Feature, Model, RunOptions, Session, StreamOptions};

use crate::Event;

/// Live streaming mode: type committed text while speaking (MYNAH_STREAM=1).
pub fn streaming_enabled() -> bool {
    std::env::var("MYNAH_STREAM").is_ok_and(|v| v == "1")
}

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
    if streaming_enabled() {
        // Streaming needs a streaming-capable model; the batch engines
        // aren't. parakeet-unified beats nemotron-3.5-asr-streaming on WER.
        return Ok(data_dir().join("parakeet-unified-en-0.6b-Q8_0.gguf"));
    }
    let engine = std::env::var("MYNAH_ENGINE").unwrap_or_else(|_| "parakeet".into());
    match engine.as_str() {
        "parakeet" => Ok(data_dir().join("parakeet-ultra-Q8_0.gguf")),
        "cohere" => Ok(data_dir().join("cohere-transcribe-03-2026-Q4_K_M.gguf")),
        // whisper.cpp .bin files load directly (backward compatible).
        "whisper" => Ok(data_dir().join("ggml-large-v3-turbo-q5_0.bin")),
        other => anyhow::bail!("unknown MYNAH_ENGINE {other:?} (use parakeet, cohere or whisper)"),
    }
}

/// MYNAH_LANG (default "en") matched to the model's language list: locale
/// models (nemotron) reject bare "en" and want "en-US", others want "en".
fn language(supported: &[String]) -> String {
    let lang = std::env::var("MYNAH_LANG").unwrap_or_else(|_| "en".into());
    if supported.is_empty() || supported.contains(&lang) {
        return lang;
    }
    let prefix = format!("{lang}-");
    supported
        .iter()
        .find(|l| l.starts_with(&prefix))
        .cloned()
        .unwrap_or(lang)
}

struct Asr {
    session: Session,
    language: String,
    vocabulary: crate::vocabulary::Vocabulary,
    /// Model accepts vocabulary terms as a decoding bias (whisper, qwen3-asr).
    vocab_bias: bool,
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
        let model =
            Model::load(&path).with_context(|| format!("loading model {}", path.display()))?;
        let caps = model.capabilities()?;
        if streaming_enabled() {
            anyhow::ensure!(
                caps.supports_streaming,
                "{} ({}) does not support streaming — MYNAH_STREAM needs a \
                 streaming-capable model (e.g. nemotron-3.5-asr-streaming)",
                path.display(),
                model.arch()
            );
        }
        let vocab_bias = model.supports(Feature::Vocabulary);
        let session = model.session().context("opening session")?;
        Ok(Self {
            session,
            language: language(&caps.languages),
            vocabulary: crate::vocabulary::Vocabulary::load()?,
            vocab_bias,
        })
    }

    /// Run options for streaming: pinned language. (Streaming models ignore
    /// runtime PNC control — punctuation is whatever the model produces.)
    fn stream_options(&self) -> RunOptions {
        RunOptions {
            language: Some(self.language.clone()),
            ..Default::default()
        }
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let options = RunOptions {
            // Pinned language (default "en") — auto-detect adds latency and
            // can misfire on short dictation clips.
            language: Some(self.language.clone()),
            vocabulary: if self.vocab_bias {
                self.vocabulary.terms()
            } else {
                Vec::new()
            },
            ..Default::default()
        };
        let transcript = self
            .session
            .run(samples, &options)
            .map_err(|e| anyhow::anyhow!("inference: {e}"))?;
        Ok(self.vocabulary.correct(transcript.text.trim()))
    }
}

enum Job {
    Batch(Vec<f32>),
    StreamStart,
    StreamChunk(Vec<f32>),
    StreamEnd,
    StreamAbort,
}

pub struct Worker {
    jobs: Sender<Job>,
}

impl Worker {
    pub fn spawn(events: Sender<Event>) -> Self {
        let (jobs, job_rx) = mpsc::channel::<Job>();

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

                while let Ok(job) = job_rx.recv() {
                    match job {
                        Job::Batch(samples) => {
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
                        Job::StreamStart => {
                            if let Err(e) = run_stream(&mut model, &job_rx, &events) {
                                log::error!("stream failed: {e:#}");
                                crate::notify(&format!("Streaming failed: {e}"));
                                events.send(Event::StreamDone).ok();
                            }
                        }
                        // Stale chunk/end from a previous session; ignore.
                        _ => {}
                    }
                }
            })
            .expect("spawning asr thread");

        Worker { jobs }
    }

    pub fn transcribe(&self, samples: Vec<f32>) {
        self.jobs
            .send(Job::Batch(samples))
            .expect("asr thread died");
    }

    pub fn stream_start(&self) {
        self.jobs.send(Job::StreamStart).expect("asr thread died");
    }

    pub fn stream_end(&self) {
        self.jobs.send(Job::StreamEnd).expect("asr thread died");
    }

    pub fn stream_abort(&self) {
        self.jobs.send(Job::StreamAbort).expect("asr thread died");
    }

    /// Sink handed to the audio capture: forwards 16 kHz chunks to the stream.
    pub fn chunk_sink(&self) -> crate::audio::ChunkSink {
        let jobs = self.jobs.clone();
        Box::new(move |chunk| {
            jobs.send(Job::StreamChunk(chunk)).ok();
        })
    }
}

/// Drive one live-streaming session: feed chunks, emit committed-text deltas.
/// Committed text is a stable prefix (it only grows), so emitting the suffix
/// since the last emit can never require correcting already-typed text.
fn run_stream(
    asr: &mut Asr,
    jobs: &mpsc::Receiver<Job>,
    events: &Sender<Event>,
) -> Result<()> {
    let options = asr.stream_options();
    let mut stream = asr
        .session
        .stream(&options, &StreamOptions::default())
        .map_err(|e| anyhow::anyhow!("opening stream: {e}"))?;
    let started = std::time::Instant::now();
    let mut emitted = 0usize;

    loop {
        match jobs.recv() {
            Ok(Job::StreamChunk(chunk)) => {
                let update = stream
                    .feed(&chunk)
                    .map_err(|e| anyhow::anyhow!("stream feed: {e}"))?;
                if update.committed_changed {
                    let committed = stream.text().committed;
                    if committed.len() > emitted {
                        events
                            .send(Event::StreamDelta(committed[emitted..].to_string()))
                            .ok();
                        emitted = committed.len();
                    }
                }
            }
            Ok(Job::StreamEnd) => {
                stream
                    .finalize()
                    .map_err(|e| anyhow::anyhow!("stream finalize: {e}"))?;
                let committed = stream.text().committed;
                if committed.len() > emitted {
                    events
                        .send(Event::StreamDelta(committed[emitted..].to_string()))
                        .ok();
                }
                log::info!("stream done in {:?}: {committed:?}", started.elapsed());
                events.send(Event::StreamDone).ok();
                return Ok(());
            }
            Ok(Job::StreamAbort) => {
                log::info!("stream aborted");
                events.send(Event::StreamDone).ok();
                return Ok(());
            }
            // Batch job or channel close mid-stream: bail out cleanly.
            Ok(Job::Batch(_)) | Ok(Job::StreamStart) => {
                events.send(Event::StreamDone).ok();
                return Ok(());
            }
            Err(_) => anyhow::bail!("job channel closed mid-stream"),
        }
    }
}

/// Print the configured model's capabilities (diagnostics for `mynah caps`).
pub fn print_caps() -> Result<()> {
    let path = model_path()?;
    let model = Model::load(&path)?;
    println!("model: {} ({})", path.display(), model.arch());
    println!("{:#?}", model.capabilities()?);
    Ok(())
}

/// File transcription for `mynah transcribe` (testing/benchmarking): loads
/// the model once, prints one transcript line per file.
pub fn transcribe_files(paths: &[String]) -> Result<()> {
    let t0 = std::time::Instant::now();
    let mut asr = Asr::load()?;
    eprintln!("load: {:.1?}", t0.elapsed());
    for path in paths {
        let samples = read_wav_16k_mono(std::path::Path::new(path))?;
        let t1 = std::time::Instant::now();
        let text = asr.transcribe(&samples)?;
        eprintln!(
            "inference: {:.3}s for {:.1}s audio",
            t1.elapsed().as_secs_f32(),
            samples.len() as f32 / 16000.0
        );
        println!("{}", text.replace('\n', " "));
    }
    Ok(())
}

/// Offline streaming test for `mynah stream-file`: feed a wav in 160ms chunks
/// as fast as possible, print committed deltas, report per-feed latency.
pub fn stream_file(path: &std::path::Path) -> Result<()> {
    let samples = read_wav_16k_mono(path)?;
    let mut asr = Asr::load()?;
    let options = asr.stream_options();
    let mut stream = asr
        .session
        .stream(&options, &StreamOptions::default())
        .map_err(|e| anyhow::anyhow!("opening stream: {e}"))?;

    let mut feed_times = Vec::new();
    let mut emitted = 0usize;
    for chunk in samples.chunks(2560) {
        let t = std::time::Instant::now();
        let update = stream
            .feed(chunk)
            .map_err(|e| anyhow::anyhow!("feed: {e}"))?;
        feed_times.push(t.elapsed());
        if update.committed_changed {
            let committed = stream.text().committed;
            if committed.len() > emitted {
                println!(
                    "[{:>5.1}s] +{:?}",
                    samples.len() as f32 / 16000.0,
                    &committed[emitted..]
                );
                emitted = committed.len();
            }
        }
    }
    stream
        .finalize()
        .map_err(|e| anyhow::anyhow!("finalize: {e}"))?;
    println!("final: {:?}", stream.text().committed);

    let avg = feed_times.iter().sum::<std::time::Duration>() / feed_times.len().max(1) as u32;
    let max = feed_times.iter().max().copied().unwrap_or_default();
    // Judge steady-state: skip the first feed (backend warmup) and require
    // p95 within the chunk budget; a rare spike only queues one chunk.
    let mut sorted: Vec<_> = feed_times.iter().skip(1).collect();
    sorted.sort();
    let p95 = sorted
        .get(sorted.len().saturating_sub(1) * 95 / 100)
        .copied()
        .copied()
        .unwrap_or_default();
    println!(
        "feeds: {} | avg {avg:?} | p95 {p95:?} | max {max:?} | budget 160ms/chunk → {}",
        feed_times.len(),
        if p95.as_millis() < 160 {
            "REALTIME OK"
        } else {
            "TOO SLOW"
        }
    );
    Ok(())
}

/// Minimal RIFF/WAVE reader: 16 kHz mono 16-bit PCM only (test tooling).
fn read_wav_16k_mono(path: &std::path::Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(
        bytes.len() > 44 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a wav file"
    );
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
