//! Bottom-of-screen recording indicator: a wlr-layer-shell surface floating
//! above all windows, drawn with tiny-skia into a shared-memory buffer.
//! Runs on its own thread with its own Wayland connection; created when
//! recording starts and torn down when the Overlay handle is dropped.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, Region},
    delegate_compositor, delegate_layer, delegate_output, delegate_registry, delegate_shm,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
        WaylandSurface,
    },
    shm::{slot::SlotPool, Shm, ShmHandler},
};
use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Rect, Transform};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_output, wl_shm, wl_surface},
    Connection, QueueHandle,
};

use crate::Level;

const WIDTH: u32 = 360;
const HEIGHT: u32 = 48;
const BOTTOM_MARGIN: i32 = 48;
const BARS: usize = 32;

pub struct Overlay {
    running: Arc<AtomicBool>,
    transcribing: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Overlay {
    pub fn spawn(level: Level) -> Result<Self> {
        let running = Arc::new(AtomicBool::new(true));
        let transcribing = Arc::new(AtomicBool::new(false));

        let r = running.clone();
        let t = transcribing.clone();
        let thread = std::thread::Builder::new()
            .name("overlay".into())
            .spawn(move || {
                if let Err(e) = run(level, r, t) {
                    log::error!("overlay: {e:#}");
                }
            })
            .context("spawning overlay thread")?;

        Ok(Self { running, transcribing, thread: Some(thread) })
    }

    pub fn set_transcribing(&self) {
        self.transcribing.store(true, Ordering::Relaxed);
    }
}

impl Drop for Overlay {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn run(level: Level, running: Arc<AtomicBool>, transcribing: Arc<AtomicBool>) -> Result<()> {
    let conn = Connection::connect_to_env().context("connecting to wayland")?;
    let (globals, queue) = registry_queue_init(&conn).context("wayland globals")?;
    let qh = queue.handle();

    let compositor = CompositorState::bind(&globals, &qh).context("wl_compositor")?;
    let layer_shell = LayerShell::bind(&globals, &qh).context("layer shell not available")?;
    let shm = Shm::bind(&globals, &qh).context("wl_shm")?;

    let surface = compositor.create_surface(&qh);
    let layer = layer_shell.create_layer_surface(&qh, surface, Layer::Overlay, Some("mynah"), None);
    layer.set_anchor(Anchor::BOTTOM);
    layer.set_size(WIDTH, HEIGHT);
    layer.set_margin(0, 0, BOTTOM_MARGIN, 0);
    layer.set_keyboard_interactivity(KeyboardInteractivity::None);

    // Click-through: empty input region so the pill never swallows clicks.
    if let Ok(region) = Region::new(&compositor) {
        layer.wl_surface().set_input_region(Some(region.wl_region()));
    }
    layer.commit();

    let pool = SlotPool::new((WIDTH * HEIGHT * 4) as usize, &shm).context("shm pool")?;

    let mut state = State {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
        shm,
        pool,
        layer,
        configured: false,
        level,
        transcribing,
        bars: VecDeque::from(vec![0.0; BARS]),
        tick: 0,
    };

    // Dispatch with a timeout so the `running` flag is honored within ~100ms
    // even if the compositor stops sending frame callbacks; otherwise
    // Overlay::drop's join could stall the main state machine.
    let mut event_loop =
        calloop::EventLoop::<State>::try_new().context("creating event loop")?;
    calloop_wayland_source::WaylandSource::new(conn.clone(), queue)
        .insert(event_loop.handle())
        .map_err(|e| anyhow::anyhow!("inserting wayland source: {e}"))?;

    while running.load(Ordering::Relaxed) {
        event_loop
            .dispatch(Some(std::time::Duration::from_millis(100)), &mut state)
            .context("wayland dispatch")?;
    }
    Ok(())
}

struct State {
    registry_state: RegistryState,
    output_state: OutputState,
    shm: Shm,
    pool: SlotPool,
    layer: LayerSurface,
    configured: bool,
    level: Level,
    transcribing: Arc<AtomicBool>,
    bars: VecDeque<f32>,
    tick: u32,
}

impl State {
    fn draw(&mut self, qh: &QueueHandle<Self>) {
        // Shift in the latest mic level; keep a rolling window of bar heights.
        let rms = f32::from_bits(self.level.load(Ordering::Relaxed));
        self.bars.pop_front();
        // Perceptual-ish scaling: mic RMS for speech is roughly 0.005–0.2.
        self.bars.push_back((rms * 12.0).powf(0.6).min(1.0));
        self.tick = self.tick.wrapping_add(1);

        let mut pixmap = Pixmap::new(WIDTH, HEIGHT).expect("pixmap");
        self.paint(&mut pixmap);

        let stride = WIDTH as i32 * 4;
        let (buffer, canvas) = self
            .pool
            .create_buffer(WIDTH as i32, HEIGHT as i32, stride, wl_shm::Format::Argb8888)
            .expect("create buffer");

        // tiny-skia is RGBA premultiplied; wl ARGB8888 little-endian is B,G,R,A.
        for (dst, src) in canvas.chunks_exact_mut(4).zip(pixmap.data().chunks_exact(4)) {
            dst[0] = src[2];
            dst[1] = src[1];
            dst[2] = src[0];
            dst[3] = src[3];
        }

        let wl_surface = self.layer.wl_surface();
        wl_surface.damage_buffer(0, 0, WIDTH as i32, HEIGHT as i32);
        wl_surface.frame(qh, wl_surface.clone());
        buffer.attach_to(wl_surface).expect("buffer attach");
        self.layer.commit();
    }

    fn paint(&self, pixmap: &mut Pixmap) {
        let w = WIDTH as f32;
        let h = HEIGHT as f32;

        // Background pill.
        let mut bg = Paint::default();
        bg.set_color(Color::from_rgba8(24, 25, 38, 235));
        bg.anti_alias = true;
        let pill = rounded_rect(0.5, 0.5, w - 1.0, h - 1.0, h / 2.0 - 0.5);
        pixmap.fill_path(&pill, &bg, FillRule::Winding, Transform::identity(), None);

        let transcribing = self.transcribing.load(Ordering::Relaxed);
        let mut fg = Paint::default();
        fg.anti_alias = true;

        if transcribing {
            // Three dots with a traveling pulse.
            fg.set_color(Color::from_rgba8(230, 175, 80, 255));
            for i in 0..3 {
                let phase = ((self.tick / 8 + i) % 3) == 2;
                let r = if phase { 5.0 } else { 3.5 };
                let cx = w / 2.0 + (i as f32 - 1.0) * 18.0;
                if let Some(c) = PathBuilder::from_circle(cx, h / 2.0, r) {
                    pixmap.fill_path(&c, &fg, FillRule::Winding, Transform::identity(), None);
                }
            }
            return;
        }

        // Red recording dot on the left.
        fg.set_color(Color::from_rgba8(247, 118, 142, 255));
        let pulse = 1.0 + 0.15 * ((self.tick as f32) / 10.0).sin();
        if let Some(dot) = PathBuilder::from_circle(h / 2.0 + 2.0, h / 2.0, 5.0 * pulse) {
            pixmap.fill_path(&dot, &fg, FillRule::Winding, Transform::identity(), None);
        }

        // Waveform bars.
        fg.set_color(Color::from_rgba8(122, 162, 247, 255));
        let left = h + 8.0;
        let right = w - h / 2.0;
        let slot = (right - left) / BARS as f32;
        let bar_w = (slot * 0.55).max(1.5);
        let max_bar = h - 18.0;
        for (i, &v) in self.bars.iter().enumerate() {
            let bh = (2.5 + v * max_bar).min(max_bar);
            let x = left + i as f32 * slot;
            let y = (h - bh) / 2.0;
            if let Some(r) = Rect::from_xywh(x, y, bar_w, bh) {
                let p = PathBuilder::from_rect(r);
                pixmap.fill_path(&p, &fg, FillRule::Winding, Transform::identity(), None);
            }
        }
    }
}

fn rounded_rect(x: f32, y: f32, w: f32, h: f32, r: f32) -> tiny_skia::Path {
    let mut pb = PathBuilder::new();
    pb.move_to(x + r, y);
    pb.line_to(x + w - r, y);
    pb.quad_to(x + w, y, x + w, y + r);
    pb.line_to(x + w, y + h - r);
    pb.quad_to(x + w, y + h, x + w - r, y + h);
    pb.line_to(x + r, y + h);
    pb.quad_to(x, y + h, x, y + h - r);
    pb.line_to(x, y + r);
    pb.quad_to(x, y, x + r, y);
    pb.close();
    pb.finish().expect("pill path")
}

impl CompositorHandler for State {
    fn scale_factor_changed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: i32) {}
    fn transform_changed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: wl_output::Transform) {}
    fn surface_enter(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: &wl_output::WlOutput) {}
    fn surface_leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: &wl_output::WlOutput) {}

    fn frame(&mut self, _: &Connection, qh: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {
        self.draw(qh);
    }
}

impl OutputHandler for State {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl LayerShellHandler for State {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {}

    fn configure(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        _: &LayerSurface,
        _configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        if !self.configured {
            self.configured = true;
            self.draw(qh);
        }
    }
}

impl ShmHandler for State {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl ProvidesRegistryState for State {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState];
}

delegate_compositor!(State);
delegate_output!(State);
delegate_shm!(State);
delegate_layer!(State);
delegate_registry!(State);
