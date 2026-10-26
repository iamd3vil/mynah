//! Model catalog and `mynah download`: pick models, fetch them into the data
//! dir with curl (resumable; renamed into place only once complete).

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result};

pub struct ModelInfo {
    /// MYNAH_ENGINE value; "streaming" is the MYNAH_STREAM=1 model.
    pub name: &'static str,
    file: &'static str,
    url: &'static str,
    size: &'static str,
    about: &'static str,
}

/// Batch engines first, in default-preference order. Accuracy/speed numbers
/// for each: scripts/bench.py and the README.
pub const MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "parakeet",
        file: "parakeet-ultra-Q8_0.gguf",
        url: "https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/main/parakeet-ultra-Q8_0.gguf",
        size: "740 MB",
        about: "fast, 25 European languages",
    },
    ModelInfo {
        name: "cohere",
        file: "cohere-transcribe-03-2026-Q4_K_M.gguf",
        url: "https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-Q4_K_M.gguf",
        size: "1.6 GB",
        about: "most accurate, ~3x slower than parakeet, 14 languages",
    },
    ModelInfo {
        name: "whisper",
        // whisper.cpp .bin files load directly (backward compatible).
        file: "ggml-large-v3-turbo-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        size: "550 MB",
        about: "slower and less accurate, but uses vocabulary.txt hints",
    },
    ModelInfo {
        name: "streaming",
        file: "parakeet-unified-en-0.6b-Q8_0.gguf",
        url: "https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q8_0.gguf",
        size: "730 MB",
        about: "types while you speak (MYNAH_STREAM=1), English only",
    },
];

impl ModelInfo {
    pub fn path(&self) -> PathBuf {
        data_dir().join(self.file)
    }
}

pub fn find(name: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.name == name)
}

fn data_dir() -> PathBuf {
    let data = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME not set")).join(".local/share")
        });
    data.join("mynah/models")
}

/// `mynah download [name...]`: the named models, or an interactive pick.
pub fn download(names: &[String]) -> Result<()> {
    let picked = if names.is_empty() {
        pick()?
    } else {
        names
            .iter()
            .map(|n| find(n).with_context(|| format!("unknown model {n:?}")))
            .collect::<Result<Vec<_>>>()?
    };
    std::fs::create_dir_all(data_dir())?;
    for model in &picked {
        fetch(model)?;
    }

    println!("\nDone. Run `mynah` (or the systemd service) and bind `mynah toggle` to a hotkey.");
    let batch: Vec<_> = MODELS
        .iter()
        .filter(|m| m.name != "streaming" && m.path().exists())
        .collect();
    if batch.len() > 1 {
        println!(
            "Engine: {} by default; set MYNAH_ENGINE={} to switch.",
            batch[0].name,
            batch[1..]
                .iter()
                .map(|m| m.name)
                .collect::<Vec<_>>()
                .join("|")
        );
    }
    if std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/uinput")
        .is_err()
    {
        println!(
            "\n/dev/uinput isn't writable, so mynah can't type yet. Allow it once with:\n\
             echo 'KERNEL==\"uinput\", GROUP=\"input\", MODE=\"0660\", TAG+=\"uaccess\"' \\\n  \
             | sudo tee /etc/udev/rules.d/60-mynah-uinput.rules\n\
             sudo udevadm control --reload && sudo udevadm trigger /dev/uinput"
        );
    }
    Ok(())
}

fn pick() -> Result<Vec<&'static ModelInfo>> {
    println!("Models (saved to {}):", data_dir().display());
    for (i, m) in MODELS.iter().enumerate() {
        let have = if m.path().exists() { "✓" } else { " " };
        println!(
            "  {have} {}) {:<10} {:>7}  {}",
            i + 1,
            m.name,
            m.size,
            m.about
        );
    }
    print!("Pick one or more, e.g. `1 2` [1]: ");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    parse_picks(&line)
}

fn parse_picks(line: &str) -> Result<Vec<&'static ModelInfo>> {
    if line.trim().is_empty() {
        return Ok(vec![&MODELS[0]]);
    }
    let mut picked: Vec<&'static ModelInfo> = Vec::new();
    for choice in line.split(|c: char| c == ',' || c.is_whitespace()) {
        if choice.is_empty() {
            continue;
        }
        let model = choice
            .parse::<usize>()
            .ok()
            .and_then(|n| MODELS.get(n.checked_sub(1)?))
            .or_else(|| find(choice))
            .with_context(|| format!("invalid choice {choice:?}"))?;
        if !picked.iter().any(|m| m.name == model.name) {
            picked.push(model);
        }
    }
    Ok(picked)
}

fn fetch(model: &ModelInfo) -> Result<()> {
    let dest = model.path();
    if dest.exists() {
        println!("{} already downloaded", model.name);
        return Ok(());
    }
    // Download beside the target and rename at the end, so an interrupted
    // download is resumed next time instead of loaded as a broken model.
    let part = dest.with_extension("part");
    println!("==> {} ({})", model.name, model.size);
    let status = Command::new("curl")
        .args(["-fL", "--retry", "3", "-C", "-", "--progress-bar", "-o"])
        .arg(&part)
        .arg(model.url)
        .status()
        .context("running curl (is it installed?)")?;
    anyhow::ensure!(
        status.success(),
        "downloading {} failed ({status}); rerun to resume",
        model.name
    );
    std::fs::rename(&part, &dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_picks;

    #[test]
    fn picks_by_number_or_name_without_duplicates() {
        let names = |line| {
            parse_picks(line)
                .unwrap()
                .iter()
                .map(|m| m.name)
                .collect::<Vec<_>>()
        };
        assert_eq!(names("\n"), ["parakeet"]);
        assert_eq!(names("2, 4 2"), ["cohere", "streaming"]);
        assert_eq!(names("whisper 1"), ["whisper", "parakeet"]);
        assert!(parse_picks("0").is_err());
        assert!(parse_picks("9").is_err());
    }
}
