//! CLI: `vectorize <image.png> [out.json]` — runs the pipeline natively and
//! writes the same JSON the Python endpoint returns. With `--stages <dir>` also
//! dumps intermediate stage outputs for parity testing.

use std::io::Write;

fn load_png(path: &str) -> (Vec<u8>, usize, usize, bool) {
    let f = std::fs::File::open(path).expect("open png");
    let decoder = png::Decoder::new(std::io::BufReader::new(f));
    let mut reader = decoder.read_info().expect("png info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("png frame");
    let (w, h) = (info.width as usize, info.height as usize);
    let bytes = &buf[..info.buffer_size()];
    let mut rgba = vec![0u8; w * h * 4];
    let has_alpha;
    match (info.color_type, info.bit_depth) {
        (png::ColorType::Rgb, png::BitDepth::Eight) => {
            for i in 0..w * h {
                rgba[4 * i] = bytes[3 * i];
                rgba[4 * i + 1] = bytes[3 * i + 1];
                rgba[4 * i + 2] = bytes[3 * i + 2];
                rgba[4 * i + 3] = 255;
            }
            has_alpha = false;
        }
        (png::ColorType::Rgba, png::BitDepth::Eight) => {
            rgba.copy_from_slice(bytes);
            has_alpha = true;
        }
        (png::ColorType::Grayscale, png::BitDepth::Eight) => {
            for i in 0..w * h {
                rgba[4 * i] = bytes[i];
                rgba[4 * i + 1] = bytes[i];
                rgba[4 * i + 2] = bytes[i];
                rgba[4 * i + 3] = 255;
            }
            has_alpha = false;
        }
        other => panic!("unsupported png format {:?}", other),
    }
    (rgba, w, h, has_alpha)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: vectorize <image.png> [out.json] [--stages <dir>] [--time]");
        std::process::exit(2);
    }
    let path = &args[1];
    let mut out_path: Option<String> = None;
    let mut stages_dir: Option<String> = None;
    let mut time = false;
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--stages" {
            stages_dir = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--time" {
            time = true;
            i += 1;
        } else {
            out_path = Some(args[i].clone());
            i += 1;
        }
    }
    let (rgba, w, h, has_alpha) = load_png(path);
    let t0 = std::time::Instant::now();
    let (result, stages) = arcline_vectorize::run_pipeline(&rgba, w, h, has_alpha);
    let dt = t0.elapsed();
    if time {
        eprintln!("{}: {:.1} ms (low={} high={})", path, dt.as_secs_f64() * 1000.0, result.low_geometry.len(), result.high_geometry.len());
    }
    let json = serde_json::to_string(&result).unwrap();
    match out_path {
        Some(p) => std::fs::write(p, json).unwrap(),
        None => {
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            lock.write_all(json.as_bytes()).unwrap();
        }
    }
    if let Some(dir) = stages_dir {
        std::fs::create_dir_all(&dir).unwrap();
        let dump_mask = |name: &str, m: &arcline_vectorize::ndimage::Mask| {
            let mut s = String::with_capacity(m.h * (m.w + 1));
            for y in 0..m.h {
                for x in 0..m.w {
                    s.push(if m.at(y, x) { '1' } else { '0' });
                }
                s.push('\n');
            }
            std::fs::write(format!("{}/{}.txt", dir, name), s).unwrap();
        };
        dump_mask("binary", &stages.binary);
        dump_mask("skeleton", &stages.skeleton);
        let polys = |v: &Vec<Vec<[f64; 2]>>| serde_json::to_string(v).unwrap();
        std::fs::write(format!("{}/raw_segments.json", dir), polys(&stages.raw_segments)).unwrap();
        std::fs::write(format!("{}/polylines.json", dir), polys(&stages.polylines)).unwrap();
        std::fs::write(format!("{}/low_unopt.json", dir), serde_json::to_string(&stages.low_unoptimized).unwrap()).unwrap();
        std::fs::write(format!("{}/high_unopt.json", dir), serde_json::to_string(&stages.high_unoptimized).unwrap()).unwrap();
        let prim_json = |v: &Vec<arcline_vectorize::lowgeo::primitives::Prim>| -> String {
            use arcline_vectorize::lowgeo::primitives::Prim;
            let items: Vec<serde_json::Value> = v
                .iter()
                .map(|p| match p {
                    Prim::Line { p0, p1 } => serde_json::json!(["L", p0, p1]),
                    Prim::Arc { p0, p1, bulge } => serde_json::json!(["A", p0, p1, bulge]),
                    Prim::Circle { center, radius } => serde_json::json!(["C", center, radius]),
                })
                .collect();
            serde_json::to_string(&items).unwrap()
        };
        std::fs::write(format!("{}/prims_initial.json", dir), prim_json(&stages.prims_initial)).unwrap();
        std::fs::write(format!("{}/prims_fitted.json", dir), prim_json(&stages.prims_fitted)).unwrap();
        std::fs::write(format!("{}/prims_consolidated.json", dir), prim_json(&stages.prims_consolidated)).unwrap();
        std::fs::write(format!("{}/tour.json", dir), serde_json::to_string(&stages.tour).unwrap()).unwrap();
    }
}
