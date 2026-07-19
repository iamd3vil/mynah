//! Microphone capture via cpal. One persistent input stream runs for the
//! daemon's lifetime, keeping a short pre-roll ring buffer so speech that
//! starts a beat before the hotkey isn't clipped; toggling recording is just
//! a flag flip (no device-open latency).
//!
//! Prefers 16 kHz mono (PipeWire resamples server-side); falls back to the
//! device default config plus a linear resample.

use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::Level;

pub const TARGET_RATE: u32 = 16_000;

/// Pre-roll kept while idle, so speech just before the hotkey is included.
const PREROLL_MS: usize = 400;

/// ~160ms at 16 kHz — the granularity live chunks are handed to the ASR.
const CHUNK: usize = 2560;

/// Receives resampled 16 kHz mono chunks during live streaming.
pub type ChunkSink = Box<dyn Fn(Vec<f32>) + Send>;

#[derive(Default)]
struct Shared {
    recording: bool,
    /// Rolling pre-roll while idle (device rate).
    ring: VecDeque<f32>,
    /// Full utterance while recording (device rate).
    buf: Vec<f32>,
    /// Samples not yet handed to the chunk sink (device rate).
    pending: Vec<f32>,
    sink: Option<ChunkSink>,
}

pub struct AudioEngine {
    // Held so the stream keeps running for the daemon's lifetime.
    _stream: cpal::Stream,
    shared: Arc<Mutex<Shared>>,
    rate: u32,
}

impl AudioEngine {
    pub fn new(level: Level) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("no default input device")?;
        log::info!("capturing from {:?}", device.description().ok());

        let (config, rate) = pick_config(&device)?;
        let channels = config.channels as usize;
        let preroll = rate as usize * PREROLL_MS / 1000;
        let chunk_at_device_rate = (CHUNK as u64 * rate as u64 / TARGET_RATE as u64) as usize;

        let shared = Arc::new(Mutex::new(Shared::default()));
        let shared2 = shared.clone();

        let stream = device
            .build_input_stream(
                config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut s = shared2.lock().unwrap();
                    let mut sum_sq = 0.0f32;
                    for frame in data.chunks_exact(channels) {
                        let mono = frame.iter().sum::<f32>() / channels as f32;
                        sum_sq += mono * mono;
                        if s.recording {
                            s.buf.push(mono);
                            if s.sink.is_some() {
                                s.pending.push(mono);
                            }
                        } else {
                            s.ring.push_back(mono);
                            if s.ring.len() > preroll {
                                s.ring.pop_front();
                            }
                        }
                    }

                    if s.recording && s.sink.is_some() {
                        while s.pending.len() >= chunk_at_device_rate {
                            let chunk: Vec<f32> = s.pending.drain(..chunk_at_device_rate).collect();
                            let chunk = if rate == TARGET_RATE {
                                chunk
                            } else {
                                resample(&chunk, rate, TARGET_RATE)
                            };
                            // Sink only forwards over a channel; safe under the lock.
                            (s.sink.as_ref().unwrap())(chunk);
                        }
                    }
                    drop(s);

                    let rms = (sum_sq / (data.len() / channels).max(1) as f32).sqrt();
                    level.store(rms.to_bits(), Ordering::Relaxed);
                },
                |e| log::error!("audio stream error: {e}"),
                None,
            )
            .context("building input stream")?;
        stream.play().context("starting input stream")?;

        Ok(Self { _stream: stream, shared, rate })
    }

    /// Begin recording: the pre-roll ring seeds the utterance buffer (and the
    /// chunk sink, in streaming mode) so already-spoken syllables are kept.
    pub fn start_recording(&self, sink: Option<ChunkSink>) {
        let mut s = self.shared.lock().unwrap();
        s.buf = s.ring.drain(..).collect();
        s.pending = if sink.is_some() { s.buf.clone() } else { Vec::new() };
        s.sink = sink;
        s.recording = true;
    }

    /// Stop recording and return the utterance as 16 kHz mono.
    pub fn stop_recording(&self) -> Vec<f32> {
        let mut s = self.shared.lock().unwrap();
        s.recording = false;
        s.sink = None;
        s.pending = Vec::new();
        let samples = std::mem::take(&mut s.buf);
        drop(s);
        log::info!(
            "captured {:.1}s of audio at {} Hz",
            samples.len() as f32 / self.rate as f32,
            self.rate
        );
        if self.rate == TARGET_RATE {
            samples
        } else {
            resample(&samples, self.rate, TARGET_RATE)
        }
    }
}

/// Prefer 16 kHz mono f32; otherwise fall back to the device default.
fn pick_config(device: &cpal::Device) -> Result<(cpal::StreamConfig, u32)> {
    if let Ok(ranges) = device.supported_input_configs() {
        for range in ranges {
            if range.sample_format() == cpal::SampleFormat::F32
                && range.channels() == 1
                && range.min_sample_rate() <= TARGET_RATE
                && range.max_sample_rate() >= TARGET_RATE
            {
                let cfg = range.with_sample_rate(TARGET_RATE).config();
                return Ok((cfg, TARGET_RATE));
            }
        }
    }
    let default = device
        .default_input_config()
        .context("no default input config")?;
    anyhow::ensure!(
        default.sample_format() == cpal::SampleFormat::F32,
        "unsupported sample format {:?}",
        default.sample_format()
    );
    let rate = default.sample_rate();
    Ok((default.config(), rate))
}

/// Linear-interpolation resampler with a boxcar pre-filter when downsampling.
/// Speech ASR is tolerant of the mild aliasing this leaves behind.
fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    // Cheap anti-alias: average over the decimation window.
    let filtered: Vec<f32> = if from > to {
        let win = (from / to).max(1) as usize;
        input
            .windows(win)
            .step_by(1)
            .map(|w| w.iter().sum::<f32>() / win as f32)
            .collect()
    } else {
        input.to_vec()
    };

    let ratio = from as f64 / to as f64;
    let out_len = (filtered.len() as f64 / ratio) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let idx = pos as usize;
        let frac = (pos - idx as f64) as f32;
        let a = filtered[idx.min(filtered.len() - 1)];
        let b = filtered[(idx + 1).min(filtered.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}
