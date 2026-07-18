//! Tray icon: a mynah bird rendered with tiny-skia, pushed to the tray as
//! SNI ARGB32 pixmaps. Mynahs are black birds with an amber beak and a bare
//! yellow patch behind the eye — that's the palette here, with a light body
//! so it reads on dark panels.

use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Transform};

use crate::Phase;

/// Render the icon for a phase at the sizes Plasma commonly asks for.
pub fn pixmaps(phase: Phase) -> Vec<ksni::Icon> {
    [24, 48].into_iter().map(|s| render(phase, s)).collect()
}

fn render(phase: Phase, size: u32) -> ksni::Icon {
    let mut pixmap = Pixmap::new(size, size).expect("icon pixmap");
    // Design coordinates are in a 100x100 box.
    let t = Transform::from_scale(size as f32 / 100.0, size as f32 / 100.0);

    let body_color = match phase {
        Phase::Loading => Color::from_rgba8(150, 150, 150, 255),
        _ => Color::from_rgba8(232, 233, 240, 255),
    };
    let amber = Color::from_rgba8(240, 168, 48, 255);
    let dark = Color::from_rgba8(32, 34, 44, 255);

    let mut paint = Paint::default();
    paint.anti_alias = true;

    let mut fill = |pm: &mut Pixmap, path: &tiny_skia::Path, color: Color| {
        let mut p = Paint::default();
        p.anti_alias = true;
        p.set_color(color);
        pm.fill_path(path, &p, FillRule::Winding, t, None);
    };

    // Tail: wedge sweeping to the lower left.
    let mut pb = PathBuilder::new();
    pb.move_to(38.0, 62.0);
    pb.line_to(4.0, 88.0);
    pb.line_to(30.0, 84.0);
    pb.close();
    fill(&mut pixmap, &pb.finish().unwrap(), body_color);

    // Body: plump ellipse, slightly tilted.
    let body = PathBuilder::from_oval(tiny_skia::Rect::from_xywh(16.0, 38.0, 56.0, 46.0).unwrap()).unwrap();
    fill(&mut pixmap, &body, body_color);

    // Head: circle overlapping the body's front.
    let head = PathBuilder::from_circle(60.0, 32.0, 20.0).unwrap();
    fill(&mut pixmap, &head, body_color);

    // Beak: amber triangle pointing right.
    let mut pb = PathBuilder::new();
    pb.move_to(76.0, 26.0);
    pb.line_to(96.0, 34.0);
    pb.line_to(76.0, 42.0);
    pb.close();
    fill(&mut pixmap, &pb.finish().unwrap(), amber);

    // Mynah's yellow eye patch: crescent behind the eye.
    let patch = PathBuilder::from_oval(tiny_skia::Rect::from_xywh(52.0, 20.0, 18.0, 12.0).unwrap()).unwrap();
    fill(&mut pixmap, &patch, amber);

    // Eye.
    let eye = PathBuilder::from_circle(64.0, 29.0, 4.5).unwrap();
    fill(&mut pixmap, &eye, dark);

    // State badge in the lower-right corner.
    let badge = match phase {
        Phase::Recording => Some(Color::from_rgba8(247, 118, 142, 255)),
        Phase::Transcribing => Some(Color::from_rgba8(230, 175, 80, 255)),
        _ => None,
    };
    if let Some(color) = badge {
        // Dark ring so the badge separates from the body.
        let ring = PathBuilder::from_circle(78.0, 78.0, 19.0).unwrap();
        fill(&mut pixmap, &ring, Color::from_rgba8(20, 21, 30, 255));
        let dot = PathBuilder::from_circle(78.0, 78.0, 13.0).unwrap();
        fill(&mut pixmap, &dot, color);
    }

    // tiny-skia gives premultiplied RGBA bytes; SNI wants ARGB32 in network
    // byte order, i.e. A,R,G,B per pixel, unpremultiplied.
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for px in pixmap.pixels() {
        let px = px.demultiply();
        data.extend_from_slice(&[px.alpha(), px.red(), px.green(), px.blue()]);
    }

    ksni::Icon { width: size as i32, height: size as i32, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render preview PNGs (dev aid): MYNAH_ICON_PREVIEW_DIR=... cargo test icon
    #[test]
    fn icon_preview() {
        let Ok(dir) = std::env::var("MYNAH_ICON_PREVIEW_DIR") else { return };
        for (phase, name) in [
            (Phase::Idle, "idle"),
            (Phase::Recording, "recording"),
            (Phase::Transcribing, "transcribing"),
            (Phase::Loading, "loading"),
        ] {
            let mut pixmap = Pixmap::new(128, 128).unwrap();
            let icon = render(phase, 128);
            for (px, chunk) in pixmap.pixels_mut().iter_mut().zip(icon.data.chunks_exact(4)) {
                *px = tiny_skia::ColorU8::from_rgba(chunk[1], chunk[2], chunk[3], chunk[0]).premultiply();
            }
            pixmap.save_png(format!("{dir}/{name}.png")).unwrap();
        }
    }

    #[test]
    fn icon_sizes_and_format() {
        let icons = pixmaps(Phase::Recording);
        assert_eq!(icons.len(), 2);
        assert_eq!(icons[0].data.len(), 24 * 24 * 4);
    }
}
