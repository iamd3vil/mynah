//! Microphone capture via cpal. Prefers 16 kHz mono (Parakeet's native rate,
//! PipeWire resamples server-side); falls back to the device default config
//! plus a linear resample on stop.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::Level;

pub const TARGET_RATE: u32 = 16_000;

/// ~160ms at 16 kHz — the granularity live chunks are handed to the ASR.
const CHUNK: usize = 2560;

/// Receives resampled 16 kHz mono chunks during live streaming.
pub type ChunkSink = Box<dyn Fn(Vec<f32>) + Send>;

pub struct Capture {
    // Held so the stream keeps running; dropped on stop.
    _stream: cpal::Stream,
    buf: Arc<Mutex<Vec<f32>>>,
    rate: u32,
}

impl Capture {
    pub fn start(level: Level, chunk_sink: Option<ChunkSink>) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("no default input device")?;
        log::info!("capturing from {:?}", device.description().ok());

        let (config, rate) = pick_config(&device)?;
        let channels = config.channels as usize;
        let buf = Arc::new(Mutex::new(Vec::<f32>::with_capacity(rate as usize * 30)));

        let buf2 = buf.clone();
        // Pending samples not yet handed to the chunk sink (device rate).
        let chunk_at_device_rate = (CHUNK as u64 * rate as u64 / TARGET_RATE as u64) as usize;
        let mut pending: Vec<f32> = Vec::with_capacity(chunk_at_device_rate * 2);
        let stream = device
            .build_input_stream(
                config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut buf = buf2.lock().unwrap();
                    let mut sum = 0.0f32;
                    let mut sum_sq = 0.0f32;
                    for frame in data.chunks_exact(channels) {
                        let mono = frame.iter().sum::<f32>() / channels as f32;
                        sum += mono;
                        sum_sq += mono * mono;
                        buf.push(mono);
                        if chunk_sink.is_some() {
                            pending.push(mono);
                        }
                    }
                    drop(buf);
                    // Some internal microphones expose a large DC offset. A raw RMS
                    // treats that offset as sound and pins the visual meter at full height,
                    // so measure the AC component (standard deviation) instead.
                    let frames = (data.len() / channels).max(1) as f32;
                    let mean = sum / frames;
                    let rms = (sum_sq / frames - mean * mean).max(0.0).sqrt();
                    level.store(rms.to_bits(), Ordering::Relaxed);

                    if let Some(sink) = &chunk_sink {
                        while pending.len() >= chunk_at_device_rate {
                            let chunk: Vec<f32> = pending.drain(..chunk_at_device_rate).collect();
                            let chunk = if rate == TARGET_RATE {
                                chunk
                            } else {
                                resample(&chunk, rate, TARGET_RATE)
                            };
                            sink(chunk);
                        }
                    }
                },
                |e| log::error!("audio stream error: {e}"),
                None,
            )
            .context("building input stream")?;
        stream.play().context("starting input stream")?;

        Ok(Self {
            _stream: stream,
            buf,
            rate,
        })
    }

    /// Stop capturing and return the samples as 16 kHz mono.
    pub fn stop(self) -> Vec<f32> {
        drop(self._stream);
        let samples = std::mem::take(&mut *self.buf.lock().unwrap());
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

#[cfg(test)]
mod tests {
    #[test]
    fn dc_offset_does_not_count_as_mic_level() {
        let samples = [0.3_f32; 480];
        let sum = samples.iter().sum::<f32>();
        let sum_sq = samples.iter().map(|sample| sample * sample).sum::<f32>();
        let frames = samples.len() as f32;
        let rms = (sum_sq / frames - (sum / frames).powi(2)).max(0.0).sqrt();

        assert!(rms < 1e-3);
    }
}
