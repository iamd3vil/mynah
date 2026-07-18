//! Unix socket control interface. A KDE global shortcut runs `mynah toggle`,
//! which connects here and sends the command to the running daemon.

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use crate::{Event, Phase};

pub fn path() -> PathBuf {
    let dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(dir).join("mynah.sock")
}

pub fn listen(tx: Sender<Event>, phase: Arc<Mutex<Phase>>) -> Result<()> {
    let sock = path();
    if sock.exists() {
        // Stale socket from a previous run, or another daemon is alive.
        if UnixStream::connect(&sock).is_ok() {
            anyhow::bail!("another mynah daemon is already running on {}", sock.display());
        }
        std::fs::remove_file(&sock)?;
    }
    let listener = UnixListener::bind(&sock).with_context(|| format!("bind {}", sock.display()))?;

    std::thread::Builder::new().name("sock".into()).spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut buf = String::new();
            if stream.read_to_string(&mut buf).is_err() {
                continue;
            }
            let reply = match buf.trim() {
                "toggle" => {
                    tx.send(Event::Toggle).ok();
                    "ok"
                }
                "cancel" => {
                    tx.send(Event::Cancel).ok();
                    "ok"
                }
                "quit" => {
                    tx.send(Event::Quit).ok();
                    "ok"
                }
                "status" => match *phase.lock().unwrap() {
                    Phase::Loading => "loading",
                    Phase::Idle => "idle",
                    Phase::Recording => "recording",
                    Phase::Transcribing => "transcribing",
                },
                _ => "unknown command",
            };
            let _ = stream.write_all(reply.as_bytes());
        }
    })?;
    Ok(())
}

/// Client side: send a command to the running daemon and print the reply.
pub fn send(cmd: &str) -> Result<()> {
    let sock = path();
    let mut stream = UnixStream::connect(&sock)
        .with_context(|| format!("is the daemon running? (socket: {})", sock.display()))?;
    stream.write_all(cmd.as_bytes())?;
    stream.shutdown(std::net::Shutdown::Write)?;
    let mut reply = String::new();
    stream.read_to_string(&mut reply)?;
    println!("{reply}");
    Ok(())
}
