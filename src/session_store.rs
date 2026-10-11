// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! Session persistence: one sidecar JSON per source folder.
//!
//! The in-memory [`Session`](crate::session::Session) is the source of truth
//! while the app runs; this module only serializes the *durable* part of it
//! (per-frame settings, LUT, preset name, retouch layer and crop) so that
//! closing the editor does not throw away a whole session's work. Undo history
//! is intentionally not persisted (it is large and not expected across
//! restarts); see `docs/ROADMAP.md` §2.4.
//!
//! Layout (deliberately **not** next to the user's images, so batch folders are
//! not polluted):
//!
//! ```text
//! config/
//!   sessions/
//!     <hash-of-folder>.json
//! ```
//!
//! The key is the *source folder* of the first image in the session, so opening
//! the same folder again restores its edits. This module has no egui/app
//! dependencies — it is a pure read/write of plain data.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::crop::Crop;
use crate::retouch::RetouchLayer;
use crate::settings::FilterSettings;

/// Schema version of a stored session. Bumped whenever the layout changes; a
/// file with a *newer* version is ignored instead of mis-parsed.
pub(crate) const SESSION_VERSION: u32 = 1;

/// The durable per-frame state.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub(crate) struct StoredFrame {
    /// Absolute or app-relative path exactly as the session addresses it.
    pub(crate) path: PathBuf,
    pub(crate) settings: FilterSettings,
    #[serde(default)]
    pub(crate) lut_path: Option<PathBuf>,
    #[serde(default)]
    pub(crate) preset_name: Option<String>,
    #[serde(default)]
    pub(crate) retouch: RetouchLayer,
    #[serde(default)]
    pub(crate) crop: Crop,
}

/// A whole session on disk.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub(crate) struct StoredSession {
    pub(crate) version: u32,
    pub(crate) frames: Vec<StoredFrame>,
    pub(crate) current_index: usize,
    /// Path of the frame the user was last on. Preferred over `current_index`
    /// when restoring, so a changed folder composition cannot point at the wrong
    /// frame. Optional for older sidecars.
    #[serde(default)]
    pub(crate) current_path: Option<PathBuf>,
}

impl Default for StoredSession {
    fn default() -> Self {
        Self {
            version: SESSION_VERSION,
            frames: Vec::new(),
            current_index: 0,
            current_path: None,
        }
    }
}

/// Folder that holds every session sidecar.
pub(crate) fn sessions_dir() -> PathBuf {
    PathBuf::from("config/sessions")
}

/// Deterministic, filesystem-safe key for a source folder.
///
/// Uses FNV-1a over a normalized spelling of the path (forward slashes, and
/// lower-case on Windows where the file system is case-insensitive) so the same
/// folder always maps to the same file across runs.
fn folder_hash(folder: &Path) -> u64 {
    let mut s = folder.to_string_lossy().replace('\\', "/");
    #[cfg(windows)]
    {
        s = s.to_lowercase();
    }
    // FNV-1a 64-bit.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Path of the sidecar for a source folder.
pub(crate) fn sidecar_path(folder: &Path) -> PathBuf {
    sessions_dir().join(format!("{:016x}.json", folder_hash(folder)))
}

/// Reads the session stored for `folder`. Returns `None` when there is no file,
/// it cannot be parsed, or it was written by a newer schema.
pub(crate) fn load(folder: &Path) -> Option<StoredSession> {
    let content = std::fs::read_to_string(sidecar_path(folder)).ok()?;
    let stored: StoredSession = serde_json::from_str(&content).ok()?;
    if stored.version > SESSION_VERSION {
        eprintln!(
            "⚠ Session store version {} is newer than supported {} — ignored",
            stored.version, SESSION_VERSION
        );
        return None;
    }
    Some(stored)
}

/// Writes a session for `folder`, creating `config/sessions/` lazily.
pub(crate) fn save(folder: &Path, session: &StoredSession) -> std::io::Result<()> {
    let dir = sessions_dir();
    std::fs::create_dir_all(&dir)?;
    let json = serde_json::to_string_pretty(session)?;
    std::fs::write(sidecar_path(folder), json)
}

/// Deletes the sidecar for `folder`, if present.
///
/// Used when a folder no longer has any edited frame (edits reverted, frames
/// removed), so a stale store cannot resurrect old edits and browsing an
/// untouched folder never leaves an empty file behind.
pub(crate) fn remove(folder: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(sidecar_path(folder)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crop::{Crop, NormRect};
    use crate::retouch::{RetouchLayer, Spot, SpotKind};

    fn sample_crop() -> Crop {
        Crop {
            rect: NormRect {
                x: 0.1,
                y: 0.2,
                w: 0.6,
                h: 0.5,
            },
            ..Crop::default()
        }
    }

    fn sample_frame(path: &str) -> StoredFrame {
        StoredFrame {
            path: PathBuf::from(path),
            settings: FilterSettings {
                exposure: 0.4,
                saturation: -12.0,
                ..FilterSettings::default()
            },
            lut_path: Some(PathBuf::from("luts/Film/Kodak.cube")),
            preset_name: Some("Portrait".to_string()),
            retouch: RetouchLayer {
                spots: vec![Spot {
                    center: [0.5, 0.5],
                    radius: 0.03,
                    hardness: 0.8,
                    opacity: 1.0,
                    kind: SpotKind::ContentAware,
                    path: Some(vec![[0.4, 0.5], [0.6, 0.5]]),
                }],
            },
            crop: sample_crop(),
        }
    }

    #[test]
    fn stored_session_round_trips_through_json() {
        let session = StoredSession {
            version: SESSION_VERSION,
            frames: vec![sample_frame("a.png"), sample_frame("b.png")],
            current_index: 1,
            current_path: Some(PathBuf::from("b.png")),
        };

        let json = serde_json::to_string(&session).expect("serialize");
        let back: StoredSession = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(session, back);
    }

    #[test]
    fn missing_optional_fields_load_as_defaults() {
        // A minimal frame written by an older build (no retouch/crop/lut fields).
        let json = r#"{
            "version": 1,
            "frames": [ { "path": "a.png", "settings": {
                "exposure": 0.0, "contrast": 0.0, "whites": 0.0, "blacks": 0.0,
                "shadows": 0.0, "highlights": 0.0, "temp": 0.0, "tint": 0.0,
                "vibrance": 0.0, "saturation": 0.0, "texture": 0.0,
                "clarity": 0.0, "dehaze": 0.0, "sharpen": 0.0,
                "lut_intensity": 0.0, "grain": 0.0
            } } ],
            "current_index": 0
        }"#;

        let stored: StoredSession = serde_json::from_str(json).expect("deserialize");
        assert_eq!(stored.frames.len(), 1);
        assert!(stored.frames[0].lut_path.is_none());
        assert!(stored.frames[0].retouch.is_empty());
        assert!(stored.frames[0].crop.is_identity());
    }

    #[test]
    fn folder_hash_is_stable_and_path_separator_agnostic() {
        let a = folder_hash(Path::new("images/set"));
        #[cfg(windows)]
        let b = folder_hash(Path::new("images\\set"));
        #[cfg(not(windows))]
        let b = folder_hash(Path::new("images/set"));
        assert_eq!(a, b);
        assert_ne!(a, folder_hash(Path::new("images/other")));
    }
}
