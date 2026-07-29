//! Plasma tray icon via StatusNotifierItem (ksni, blocking backend).

use std::sync::mpsc::Sender;

use anyhow::Result;
use ksni::blocking::TrayMethods;

use crate::{Event, Phase};

struct MynahTray {
    phase: Phase,
    tx: Sender<Event>,
}

impl ksni::Tray for MynahTray {
    fn id(&self) -> String {
        "mynah".into()
    }

    fn title(&self) -> String {
        format!("mynah — {}", phase_label(self.phase))
    }

    // No icon_name: with it empty, hosts fall back to our pixmaps.
    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        crate::icon::pixmaps(self.phase)
    }

    fn tool_tip(&self) -> ksni::ToolTip {
        ksni::ToolTip {
            title: "mynah".into(),
            description: phase_label(self.phase).into(),
            ..Default::default()
        }
    }

    // Left click on the icon toggles dictation.
    fn activate(&mut self, _x: i32, _y: i32) {
        self.tx.send(Event::Toggle).ok();
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::*;
        vec![
            StandardItem {
                label: match self.phase {
                    Phase::Recording => "Stop & type",
                    _ => "Start dictation",
                }
                .into(),
                icon_name: "audio-input-microphone".into(),
                activate: Box::new(|this: &mut Self| {
                    this.tx.send(Event::Toggle).ok();
                }),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: "Cancel recording".into(),
                icon_name: "process-stop".into(),
                activate: Box::new(|this: &mut Self| {
                    this.tx.send(Event::Cancel).ok();
                }),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: "Quit".into(),
                icon_name: "application-exit".into(),
                activate: Box::new(|this: &mut Self| {
                    this.tx.send(Event::Quit).ok();
                }),
                ..Default::default()
            }
            .into(),
        ]
    }
}

fn phase_label(phase: Phase) -> &'static str {
    match phase {
        Phase::Loading => "loading model…",
        Phase::Idle => "idle",
        Phase::Recording => "recording",
        Phase::Transcribing => "transcribing…",
    }
}

pub struct Tray {
    handle: ksni::blocking::Handle<MynahTray>,
}

impl Tray {
    pub fn set_phase(&self, phase: Phase) {
        self.handle.update(|tray| tray.phase = phase);
    }
}

pub fn spawn(tx: Sender<Event>) -> Result<Tray> {
    let handle = MynahTray {
        phase: Phase::Loading,
        tx,
    }
    .spawn()
    .map_err(|e| anyhow::anyhow!("tray: {e}"))?;
    Ok(Tray { handle })
}
