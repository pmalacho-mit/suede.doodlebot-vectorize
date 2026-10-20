//! Drawing command type shared by all stages.

use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Command {
    Line { distance: f64, #[serde(rename = "penDown")] pen_down: bool },
    Spin { degrees: f64 },
    Arc { radius: f64, degrees: f64 },
}
