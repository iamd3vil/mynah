//! Raw keystroke injection via a uinput virtual keyboard. Works in every
//! Wayland app including terminals, because KWin sees a real keyboard.
//!
//! Keycodes assume a US layout; text is normalized to ASCII first (dictation
//! output is ASCII apart from the occasional typographic character).

use std::thread::sleep;
use std::time::Duration;

use anyhow::{Context, Result};
use evdev::uinput::VirtualDevice;
use evdev::{AttributeSet, KeyCode, KeyEvent};

/// Pause between key events so slow clients (Electron…) don't drop any.
const KEY_DELAY: Duration = Duration::from_millis(2);

pub struct Typer {
    dev: VirtualDevice,
}

impl Typer {
    pub fn new() -> Result<Self> {
        let mut keys = AttributeSet::<KeyCode>::new();
        for code in ALL_KEYS {
            keys.insert(*code);
        }
        let dev = VirtualDevice::builder()
            .context("opening /dev/uinput (is the udev ACL/input group set up?)")?
            .name("mynah virtual keyboard")
            .with_keys(&keys)
            .context("registering keys")?
            .build()
            .context("creating virtual keyboard")?;
        Ok(Self { dev })
    }

    pub fn type_text(&mut self, text: &str) -> Result<()> {
        // Let the compositor settle focus after the overlay closes.
        sleep(Duration::from_millis(60));
        self.type_delta(text)
    }

    /// Type without the settle delay — for live streaming deltas.
    pub fn type_delta(&mut self, text: &str) -> Result<()> {
        for ch in text.chars().flat_map(normalize) {
            let Some((code, shift)) = keymap(ch) else {
                log::warn!("no keycode for {ch:?}, skipping");
                continue;
            };
            if shift {
                self.dev
                    .emit(&[*KeyEvent::new(KeyCode::KEY_LEFTSHIFT, 1)])?;
            }
            self.dev.emit(&[*KeyEvent::new(code, 1)])?;
            self.dev.emit(&[*KeyEvent::new(code, 0)])?;
            if shift {
                self.dev
                    .emit(&[*KeyEvent::new(KeyCode::KEY_LEFTSHIFT, 0)])?;
            }
            sleep(KEY_DELAY);
        }
        Ok(())
    }
}

/// Replace common typographic characters with ASCII equivalents.
fn normalize(ch: char) -> impl Iterator<Item = char> {
    let s: &'static str = match ch {
        '\u{2018}' | '\u{2019}' | '\u{02BC}' => "'",
        '\u{201C}' | '\u{201D}' => "\"",
        '\u{2013}' | '\u{2014}' | '\u{2212}' => "-",
        '\u{2026}' => "...",
        '\u{00A0}' | '\u{2009}' | '\u{200A}' => " ",
        _ => return Normalized::Same(std::iter::once(ch)),
    };
    Normalized::Replaced(s.chars())
}

enum Normalized {
    Same(std::iter::Once<char>),
    Replaced(std::str::Chars<'static>),
}

impl Iterator for Normalized {
    type Item = char;
    fn next(&mut self) -> Option<char> {
        match self {
            Normalized::Same(i) => i.next(),
            Normalized::Replaced(i) => i.next(),
        }
    }
}

/// US-layout keymap: char -> (keycode, needs shift).
fn keymap(ch: char) -> Option<(KeyCode, bool)> {
    use KeyCode as K;
    let (code, shift) = match ch {
        'a'..='z' => (letter(ch), false),
        'A'..='Z' => (letter(ch.to_ascii_lowercase()), true),
        '1' => (K::KEY_1, false),
        '2' => (K::KEY_2, false),
        '3' => (K::KEY_3, false),
        '4' => (K::KEY_4, false),
        '5' => (K::KEY_5, false),
        '6' => (K::KEY_6, false),
        '7' => (K::KEY_7, false),
        '8' => (K::KEY_8, false),
        '9' => (K::KEY_9, false),
        '0' => (K::KEY_0, false),
        '!' => (K::KEY_1, true),
        '@' => (K::KEY_2, true),
        '#' => (K::KEY_3, true),
        '$' => (K::KEY_4, true),
        '%' => (K::KEY_5, true),
        '^' => (K::KEY_6, true),
        '&' => (K::KEY_7, true),
        '*' => (K::KEY_8, true),
        '(' => (K::KEY_9, true),
        ')' => (K::KEY_0, true),
        ' ' => (K::KEY_SPACE, false),
        '\n' => (K::KEY_ENTER, false),
        '\t' => (K::KEY_TAB, false),
        '-' => (K::KEY_MINUS, false),
        '_' => (K::KEY_MINUS, true),
        '=' => (K::KEY_EQUAL, false),
        '+' => (K::KEY_EQUAL, true),
        '[' => (K::KEY_LEFTBRACE, false),
        '{' => (K::KEY_LEFTBRACE, true),
        ']' => (K::KEY_RIGHTBRACE, false),
        '}' => (K::KEY_RIGHTBRACE, true),
        '\\' => (K::KEY_BACKSLASH, false),
        '|' => (K::KEY_BACKSLASH, true),
        ';' => (K::KEY_SEMICOLON, false),
        ':' => (K::KEY_SEMICOLON, true),
        '\'' => (K::KEY_APOSTROPHE, false),
        '"' => (K::KEY_APOSTROPHE, true),
        ',' => (K::KEY_COMMA, false),
        '<' => (K::KEY_COMMA, true),
        '.' => (K::KEY_DOT, false),
        '>' => (K::KEY_DOT, true),
        '/' => (K::KEY_SLASH, false),
        '?' => (K::KEY_SLASH, true),
        '`' => (K::KEY_GRAVE, false),
        '~' => (K::KEY_GRAVE, true),
        _ => return None,
    };
    Some((code, shift))
}

fn letter(ch: char) -> KeyCode {
    use KeyCode as K;
    match ch {
        'a' => K::KEY_A,
        'b' => K::KEY_B,
        'c' => K::KEY_C,
        'd' => K::KEY_D,
        'e' => K::KEY_E,
        'f' => K::KEY_F,
        'g' => K::KEY_G,
        'h' => K::KEY_H,
        'i' => K::KEY_I,
        'j' => K::KEY_J,
        'k' => K::KEY_K,
        'l' => K::KEY_L,
        'm' => K::KEY_M,
        'n' => K::KEY_N,
        'o' => K::KEY_O,
        'p' => K::KEY_P,
        'q' => K::KEY_Q,
        'r' => K::KEY_R,
        's' => K::KEY_S,
        't' => K::KEY_T,
        'u' => K::KEY_U,
        'v' => K::KEY_V,
        'w' => K::KEY_W,
        'x' => K::KEY_X,
        'y' => K::KEY_Y,
        'z' => K::KEY_Z,
        _ => unreachable!(),
    }
}

const ALL_KEYS: &[KeyCode] = &[
    KeyCode::KEY_A,
    KeyCode::KEY_B,
    KeyCode::KEY_C,
    KeyCode::KEY_D,
    KeyCode::KEY_E,
    KeyCode::KEY_F,
    KeyCode::KEY_G,
    KeyCode::KEY_H,
    KeyCode::KEY_I,
    KeyCode::KEY_J,
    KeyCode::KEY_K,
    KeyCode::KEY_L,
    KeyCode::KEY_M,
    KeyCode::KEY_N,
    KeyCode::KEY_O,
    KeyCode::KEY_P,
    KeyCode::KEY_Q,
    KeyCode::KEY_R,
    KeyCode::KEY_S,
    KeyCode::KEY_T,
    KeyCode::KEY_U,
    KeyCode::KEY_V,
    KeyCode::KEY_W,
    KeyCode::KEY_X,
    KeyCode::KEY_Y,
    KeyCode::KEY_Z,
    KeyCode::KEY_1,
    KeyCode::KEY_2,
    KeyCode::KEY_3,
    KeyCode::KEY_4,
    KeyCode::KEY_5,
    KeyCode::KEY_6,
    KeyCode::KEY_7,
    KeyCode::KEY_8,
    KeyCode::KEY_9,
    KeyCode::KEY_0,
    KeyCode::KEY_SPACE,
    KeyCode::KEY_ENTER,
    KeyCode::KEY_TAB,
    KeyCode::KEY_MINUS,
    KeyCode::KEY_EQUAL,
    KeyCode::KEY_LEFTBRACE,
    KeyCode::KEY_RIGHTBRACE,
    KeyCode::KEY_BACKSLASH,
    KeyCode::KEY_SEMICOLON,
    KeyCode::KEY_APOSTROPHE,
    KeyCode::KEY_COMMA,
    KeyCode::KEY_DOT,
    KeyCode::KEY_SLASH,
    KeyCode::KEY_GRAVE,
    KeyCode::KEY_LEFTSHIFT,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keymap_covers_printable_ascii() {
        for b in 0x20u8..=0x7e {
            let ch = b as char;
            assert!(keymap(ch).is_some(), "no mapping for {ch:?}");
        }
    }

    #[test]
    fn normalize_typographic() {
        let s: String = "\u{201C}hi\u{201D}\u{2014}ok\u{2026}"
            .chars()
            .flat_map(normalize)
            .collect();
        assert_eq!(s, "\"hi\"-ok...");
    }
}
