//! Brief audible cues for recording state changes.

use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// The two recording-state cues.
#[derive(Clone, Copy)]
pub enum Cue {
    Start,
    Stop,
}

/// Play a short cue on the default output device.
///
/// Sound is enabled by default. Set `MYNAH_SOUNDS=0` to disable it. Failures
/// are deliberately non-fatal: dictation must still work without speakers.
pub fn play(cue: Cue) {
    if sounds_disabled() {
        return;
    }
    if let Err(e) = play_tone(cue) {
        log::debug!("could not play recording cue: {e:#}");
    }
}

fn sounds_disabled() -> bool {
    matches!(
        std::env::var("MYNAH_SOUNDS").ok().as_deref(),
        Some("0" | "false" | "off")
    )
}

fn play_tone(cue: Cue) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("no default output device"))?;
    let config = device
        .supported_output_configs()?
        .find(|range| range.sample_format() == cpal::SampleFormat::F32)
        .map(|range| range.with_max_sample_rate().config())
        .ok_or_else(|| anyhow::anyhow!("output device has no f32 format"))?;

    let rate = config.sample_rate;
    let channels = config.channels as usize;
    let (frequency, duration) = match cue {
        Cue::Start => (880.0, Duration::from_millis(65)),
        Cue::Stop => (660.0, Duration::from_millis(85)),
    };
    let total_frames = (rate as f64 * duration.as_secs_f64()) as usize;
    let mut frame = 0usize;
    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for output_frame in data.chunks_mut(channels) {
                let sample = if frame >= total_frames {
                    0.0
                } else {
                    let progress = frame as f32 / total_frames as f32;
                    // A short fade prevents clicks at either edge of the tone.
                    let envelope = (progress / 0.15).min((1.0 - progress) / 0.15).min(1.0);
                    (2.0 * std::f32::consts::PI * frequency * frame as f32 / rate as f32).sin()
                        * envelope
                        * 0.12
                };
                output_frame.fill(sample);
                frame += 1;
            }
        },
        |e| log::debug!("recording cue stream error: {e}"),
        None,
    )?;
    stream.play()?;
    std::thread::sleep(duration + Duration::from_millis(25));
    Ok(())
}
