//! Parakeet transcription worker. Owns the model on a dedicated thread so
//! inference never blocks the main state machine.

use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};

use anyhow::{Context, Result};
use transcribe_rs::onnx::parakeet::ParakeetModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::{SpeechModel, TranscribeOptions};

use crate::Event;

pub fn model_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MYNAH_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    let data = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME not set")).join(".local/share")
        });
    data.join("mynah/models/parakeet-tdt-0.6b-v3-int8")
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
                let dir = model_dir();
                log::info!("loading parakeet model from {}", dir.display());
                let mut model = match load(&dir) {
                    Ok(m) => {
                        events.send(Event::ModelReady).ok();
                        m
                    }
                    Err(e) => {
                        log::error!("model load failed: {e:#}");
                        crate::notify(&format!(
                            "Model load failed: {e}\nExpected model at {}",
                            dir.display()
                        ));
                        return;
                    }
                };

                while let Ok(samples) = job_rx.recv() {
                    let started = std::time::Instant::now();
                    let result = model
                        .transcribe(&samples, &TranscribeOptions::default())
                        .map(|r| r.text)
                        .map_err(anyhow::Error::from);
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
    let mut model = load(&model_dir())?;
    let result = model.transcribe_file(path, &TranscribeOptions::default())?;
    Ok(result.text)
}

fn load(dir: &std::path::Path) -> Result<ParakeetModel> {
    anyhow::ensure!(
        dir.join("vocab.txt").exists(),
        "model directory missing (run scripts/download-model.sh)"
    );
    ParakeetModel::load(dir, &Quantization::Int8).context("loading parakeet model")
}
