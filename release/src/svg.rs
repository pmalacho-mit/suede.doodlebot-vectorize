//! SVG rendering (`visualize.commands_to_svg` / `commands_to_svg_compare`).

use crate::fmath::DetMath;
use crate::commands::Command;
use crate::numeric::P2;
use std::fmt::Write;

enum Drawn {
    Line { p0: P2, p1: P2 },
    Arc { p0: P2, p1: P2, center: P2, radius: f64, sweep: f64 },
}

struct Sim {
    drawn: Vec<Drawn>,
    pen_up: Vec<(P2, P2)>,
    bbox: (f64, f64, f64, f64),
}

fn simulate(cmds: &[Command], start_pos: P2, start_heading: f64) -> Sim {
    let mut pos = start_pos;
    let mut heading = start_heading;
    let mut drawn = Vec::new();
    let mut pen_up = Vec::new();
    let (mut minx, mut miny, mut maxx, mut maxy) = (pos[0], pos[1], pos[0], pos[1]);
    let mut upd = |x: f64, y: f64| {
        if x < minx {
            minx = x;
        }
        if y < miny {
            miny = y;
        }
        if x > maxx {
            maxx = x;
        }
        if y > maxy {
            maxy = y;
        }
    };
    for cmd in cmds {
        match cmd {
            Command::Spin { degrees } => heading += degrees.to_radians(),
            Command::Line { distance, pen_down } => {
                let new_pos = [pos[0] + distance * heading.lcos(), pos[1] + distance * heading.lsin()];
                if *pen_down {
                    drawn.push(Drawn::Line { p0: pos, p1: new_pos });
                } else {
                    pen_up.push((pos, new_pos));
                }
                upd(new_pos[0], new_pos[1]);
                pos = new_pos;
            }
            Command::Arc { radius, degrees } => {
                let r = *radius;
                let sweep = degrees.to_radians();
                let ccw = sweep > 0.0;
                let normal_angle = heading + if ccw { std::f64::consts::PI / 2.0 } else { -std::f64::consts::PI / 2.0 };
                let center = [pos[0] + r * normal_angle.lcos(), pos[1] + r * normal_angle.lsin()];
                let start_a = (pos[1] - center[1]).latan2(pos[0] - center[0]);
                let end_a = start_a + sweep;
                let new_pos = [center[0] + r * end_a.lcos(), center[1] + r * end_a.lsin()];
                drawn.push(Drawn::Arc { p0: pos, p1: new_pos, center, radius: r, sweep });
                let n_samp = 2usize.max((sweep.abs() * 8.0) as i64 as usize);
                for k in 0..=n_samp {
                    let t = k as f64 / n_samp as f64;
                    let a = start_a + t * sweep;
                    upd(center[0] + r * a.lcos(), center[1] + r * a.lsin());
                }
                pos = new_pos;
                heading += sweep;
            }
        }
    }
    Sim { drawn, pen_up, bbox: (minx, miny, maxx, maxy) }
}

/// Python's `f"{x:.{p}f}"` (round-half-even on the exact binary value).
fn pf(x: f64, p: usize) -> String {
    let s = format!("{:.*}", p, x);
    // Rust prints "-0.00" for tiny negatives just like Python does.
    s
}

fn hsl(hue: f64) -> String {
    format!("hsl({}, 80%, 50%)", pf(hue, 1))
}

fn render_parts(sim: &Sim, stroke_width: &str, pen_up_stroke_width: &str, out: &mut Vec<String>) {
    out.push("<g stroke=\"#bbb\" stroke-dasharray=\"2,2\" fill=\"none\">".to_string());
    for (p0, p1) in &sim.pen_up {
        out.push(format!(
            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke-width=\"{}\" />",
            pf(p0[0], 2),
            pf(p0[1], 2),
            pf(p1[0], 2),
            pf(p1[1], 2),
            pen_up_stroke_width
        ));
    }
    out.push("</g>".to_string());
    let n = sim.drawn.len();
    out.push("<g fill=\"none\" stroke-linecap=\"round\">".to_string());
    for (i, d) in sim.drawn.iter().enumerate() {
        let color = hsl(360.0 * i as f64 / (n.max(1)) as f64);
        match d {
            Drawn::Line { p0, p1 } => out.push(format!(
                "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\" />",
                pf(p0[0], 2),
                pf(p0[1], 2),
                pf(p1[0], 2),
                pf(p1[1], 2),
                color,
                stroke_width
            )),
            Drawn::Arc { p0, p1, center, radius, sweep } => {
                let r = *radius;
                if sweep.abs() >= 2.0 * std::f64::consts::PI - 1e-3 {
                    let start_a = (p0[1] - center[1]).latan2(p0[0] - center[0]);
                    let mid_a = start_a + sweep / 2.0;
                    let pmx = center[0] + r * mid_a.lcos();
                    let pmy = center[1] + r * mid_a.lsin();
                    let flag = if *sweep > 0.0 { 1 } else { 0 };
                    out.push(format!(
                        "  <path d=\"M {} {} A {} {} 0 0 {} {} {} A {} {} 0 0 {} {} {}\" stroke=\"{}\" stroke-width=\"{}\" />",
                        pf(p0[0], 2),
                        pf(p0[1], 2),
                        pf(r, 2),
                        pf(r, 2),
                        flag,
                        pf(pmx, 2),
                        pf(pmy, 2),
                        pf(r, 2),
                        pf(r, 2),
                        flag,
                        pf(p1[0], 2),
                        pf(p1[1], 2),
                        color,
                        stroke_width
                    ));
                } else {
                    let large = if sweep.abs() > std::f64::consts::PI { 1 } else { 0 };
                    let flag = if *sweep > 0.0 { 1 } else { 0 };
                    out.push(format!(
                        "  <path d=\"M {} {} A {} {} 0 {} {} {} {}\" stroke=\"{}\" stroke-width=\"{}\" />",
                        pf(p0[0], 2),
                        pf(p0[1], 2),
                        pf(r, 2),
                        pf(r, 2),
                        large,
                        flag,
                        pf(p1[0], 2),
                        pf(p1[1], 2),
                        color,
                        stroke_width
                    ));
                }
            }
        }
    }
    out.push("</g>".to_string());
    if n > 0 {
        let first = match &sim.drawn[0] {
            Drawn::Line { p0, .. } | Drawn::Arc { p0, .. } => *p0,
        };
        let last = match &sim.drawn[n - 1] {
            Drawn::Line { p1, .. } | Drawn::Arc { p1, .. } => *p1,
        };
        out.push(format!("<circle cx=\"{}\" cy=\"{}\" r=\"2\" fill=\"black\" />", pf(first[0], 2), pf(first[1], 2)));
        out.push(format!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"3\" fill=\"none\" stroke=\"black\" stroke-width=\"1\" />",
            pf(last[0], 2),
            pf(last[1], 2)
        ));
    }
}

pub fn commands_to_svg(cmds: &[Command]) -> String {
    let padding = 8.0;
    let sim = simulate(cmds, [0.0, 0.0], 0.0);
    let (mut minx, mut miny, mut maxx, mut maxy) = sim.bbox;
    minx -= padding;
    miny -= padding;
    maxx += padding;
    maxy += padding;
    let width = maxx - minx;
    let height = maxy - miny;
    let mut parts = vec![
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{} {} {} {}\" width=\"{}\" height=\"{}\">",
            pf(minx, 2),
            pf(miny, 2),
            pf(width, 2),
            pf(height, 2),
            pf(width, 0),
            pf(height, 0)
        ),
        format!("<rect width=\"100%\" height=\"100%\" fill=\"white\" x=\"{}\" y=\"{}\" />", pf(minx, 2), pf(miny, 2)),
    ];
    render_parts(&sim, "1.5", "0.5", &mut parts);
    parts.push("</svg>".to_string());
    parts.join("\n")
}

pub fn commands_to_svg_compare(a: &[Command], b: &[Command], label_a: &str, label_b: &str) -> String {
    let padding = 8.0;
    let panel_gap = 24.0;
    let label_height = 28.0;
    let sa = simulate(a, [0.0, 0.0], 0.0);
    let sb = simulate(b, [0.0, 0.0], 0.0);
    let minx = sa.bbox.0.min(sb.bbox.0) - padding;
    let miny = sa.bbox.1.min(sb.bbox.1) - padding;
    let maxx = sa.bbox.2.max(sb.bbox.2) + padding;
    let maxy = sa.bbox.3.max(sb.bbox.3) + padding;
    let panel_w = maxx - minx;
    let panel_h = maxy - miny;
    let total_w = 2.0 * panel_w + panel_gap;
    let total_h = panel_h + label_height;
    let label_baseline = label_height * 0.7;
    let font_size = label_height * 0.5;
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
        pf(total_w, 2),
        pf(total_h, 2),
        pf(total_w, 0),
        pf(total_h, 0)
    ));
    parts.push("<rect width=\"100%\" height=\"100%\" fill=\"white\" />".to_string());
    parts.push(format!(
        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#e0e0e0\" stroke-width=\"0.5\" />",
        pf(panel_w + panel_gap / 2.0, 2),
        pf(label_height, 2),
        pf(panel_w + panel_gap / 2.0, 2),
        pf(total_h, 2)
    ));
    parts.push(format!(
        "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"{}\" fill=\"#333\">{}</text>",
        pf(panel_w / 2.0, 2),
        pf(label_baseline, 2),
        pf(font_size, 2),
        label_a
    ));
    parts.push(format!(
        "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"{}\" fill=\"#333\">{}</text>",
        pf(panel_w + panel_gap + panel_w / 2.0, 2),
        pf(label_baseline, 2),
        pf(font_size, 2),
        label_b
    ));
    parts.push(format!("<g transform=\"translate({}, {})\">", pf(-minx, 2), pf(label_height - miny, 2)));
    render_parts(&sa, "1.5", "0.5", &mut parts);
    parts.push("</g>".to_string());
    parts.push(format!("<g transform=\"translate({}, {})\">", pf(panel_w + panel_gap - minx, 2), pf(label_height - miny, 2)));
    render_parts(&sb, "1.5", "0.5", &mut parts);
    parts.push("</g>".to_string());
    parts.push("</svg>".to_string());
    let mut s = String::new();
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            s.push('\n');
        }
        let _ = write!(s, "{}", p);
    }
    s
}
