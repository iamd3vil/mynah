//! User-defined post-transcription corrections and Whisper vocabulary hints.

use std::path::PathBuf;

use anyhow::{Context, Result};

#[derive(Debug)]
struct Rule {
    heard: String,
    replacement: String,
}

#[derive(Debug, Default)]
pub struct Vocabulary {
    rules: Vec<Rule>,
}

impl Vocabulary {
    pub fn load() -> Result<Self> {
        let path = config_path();
        if !path.exists() {
            log::info!("no vocabulary file at {}", path.display());
            return Ok(Self::default());
        }

        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading vocabulary file {}", path.display()))?;
        let mut rules = Vec::new();
        for (line_number, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (heard, replacement) = line.split_once("=>").with_context(|| {
                format!(
                    "invalid vocabulary entry on line {} (expected `misheard => desired`)",
                    line_number + 1
                )
            })?;
            let heard = heard.trim();
            let replacement = replacement.trim();
            anyhow::ensure!(
                !heard.is_empty() && !replacement.is_empty(),
                "invalid vocabulary entry on line {} (both sides are required)",
                line_number + 1
            );
            rules.push(Rule {
                heard: heard.into(),
                replacement: replacement.into(),
            });
        }

        // Prefer the most specific phrase when rules overlap.
        rules.sort_by_key(|rule| std::cmp::Reverse(rule.heard.len()));
        log::info!(
            "loaded {} vocabulary correction(s) from {}",
            rules.len(),
            path.display()
        );
        Ok(Self { rules })
    }

    /// Correct complete words and phrases while preserving the desired spelling.
    pub fn correct(&self, text: &str) -> String {
        self.rules.iter().fold(text.to_owned(), |text, rule| {
            replace_whole_phrase(&text, &rule.heard, &rule.replacement)
        })
    }

    /// A contextual bias for Whisper. Parakeet currently has no equivalent API.
    pub fn whisper_prompt(&self) -> Option<String> {
        if self.rules.is_empty() {
            return None;
        }
        let terms = self
            .rules
            .iter()
            .map(|rule| rule.replacement.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!("Technical terms and names: {terms}."))
    }
}

fn config_path() -> PathBuf {
    let config = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME not set")).join(".config")
        });
    config.join("mynah/vocabulary.txt")
}

fn replace_whole_phrase(text: &str, heard: &str, replacement: &str) -> String {
    // ASR output is normally lower-case, but matching ASCII case-insensitively
    // also handles sentence starts without changing UTF-8 byte offsets.
    let comparable = text.to_ascii_lowercase();
    let heard = heard.to_ascii_lowercase();
    let mut result = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some(relative_start) = comparable[cursor..].find(&heard) {
        let start = cursor + relative_start;
        let end = start + heard.len();
        if is_boundary_before(text, start) && is_boundary_after(text, end) {
            result.push_str(&text[cursor..start]);
            result.push_str(replacement);
            cursor = end;
        } else {
            let next = start + heard.len();
            result.push_str(&text[cursor..next]);
            cursor = next;
        }
    }
    result.push_str(&text[cursor..]);
    result
}

fn is_boundary_before(text: &str, index: usize) -> bool {
    text[..index]
        .chars()
        .next_back()
        .is_none_or(|ch| !ch.is_alphanumeric())
}

fn is_boundary_after(text: &str, index: usize) -> bool {
    text[index..]
        .chars()
        .next()
        .is_none_or(|ch| !ch.is_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::{Rule, Vocabulary};

    #[test]
    fn replaces_whole_phrases_but_not_substrings() {
        let vocabulary = Vocabulary {
            rules: vec![Rule {
                heard: "rust".into(),
                replacement: "Rust".into(),
            }],
        };
        assert_eq!(vocabulary.correct("rust is trusty"), "Rust is trusty");
    }

    #[test]
    fn longer_phrases_win() {
        let vocabulary = Vocabulary {
            rules: vec![
                Rule {
                    heard: "kite connect".into(),
                    replacement: "Kite Connect".into(),
                },
                Rule {
                    heard: "kite".into(),
                    replacement: "Kite".into(),
                },
            ],
        };
        assert_eq!(vocabulary.correct("kite connect"), "Kite Connect");
    }
}
