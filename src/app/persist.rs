// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! Session persistence glue (P0-4).
//!
//! Thin app layer on top of the pure [`crate::session_store`]: it flushes the
//! live frame into the in-memory session, serializes the durable per-frame state
//! to `config/sessions/<hash>.json`, and restores it when a folder is reopened.
//! Autosave is debounced by the caller (the `update` loop) via
//! [`TinyLumaApp::mark_session_dirty`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::TinyLumaApp;
use crate::session_store::{self, SESSION_VERSION, StoredFrame, StoredSession};
use crate::ui::notification::ToastKind;

/// Canonical key for path matching. Falls back to the raw path when the file
/// cannot be canonicalized (e.g. it is missing), so matching degrades gracefully
/// instead of failing.
fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

impl TinyLumaApp {
    /// Reads the resume pointer from the config and validates it against the
    /// sidecar. Returns `(folder, edits)` only when the folder still exists and
    /// the sidecar holds at least one edit — otherwise the hero button would be
    /// a dead end.
    pub(crate) fn load_last_session(configured: Option<String>) -> (Option<PathBuf>, usize) {
        let Some(folder) = configured.filter(|s| !s.is_empty()).map(PathBuf::from) else {
            return (None, 0);
        };
        if !folder.is_dir() {
            return (None, 0);
        }
        let edits = session_store::load(&folder)
            .map(|s| s.frames.len())
            .unwrap_or(0);
        if edits == 0 {
            (None, 0)
        } else {
            (Some(folder), edits)
        }
    }

    /// Source folder that keys the persisted session: the parent of the first
    /// image in the filmstrip.
    pub(crate) fn session_folder(&self) -> Option<PathBuf> {
        self.session
            .image_list
            .first()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
    }

    /// `(folder name, edit count)` for the hero "Continue last session" button,
    /// or `None` when there is nothing to resume.
    pub(crate) fn resume_label(&self) -> Option<(String, usize)> {
        let folder = self.last_session_folder.as_ref()?;
        if self.last_session_edits == 0 {
            return None;
        }
        let name = folder
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| folder.to_string_lossy().into_owned());
        Some((name, self.last_session_edits))
    }

    /// Flags that the session has unsaved changes and (re)starts the debounce.
    pub(crate) fn mark_session_dirty(&mut self) {
        self.session_dirty = true;
        self.last_session_change = std::time::Instant::now();
    }

    /// Writes the current live frame into the per-file session maps. Called
    /// before persisting so a just-made edit is not lost (the maps are normally
    /// synced at gesture completion, but the debounced autosave can fire between
    /// edge cases).
    fn flush_current_frame(&mut self) {
        if let Some(path) = self.image_path.clone() {
            let preset_name = self.current_preset_name();
            self.session.save_current_settings(
                &path,
                self.settings,
                self.lut_path.clone(),
                preset_name,
            );
            let layer = self.retouch.layer.clone();
            self.session.save_retouch(&path, layer);
            self.session.save_crop(&path, self.crop.crop);
        }
    }

    /// Serializes the session to its sidecar. Only modified frames are stored,
    /// so an untouched folder keeps no file at all.
    pub(crate) fn persist_session(&mut self) {
        let Some(folder) = self.session_folder() else {
            return;
        };
        self.flush_current_frame();

        let frames: Vec<StoredFrame> = self
            .session
            .image_list
            .iter()
            .filter(|path| self.session.is_modified(path))
            .map(|path| {
                let (settings, lut_path, preset_name) = self.session.load_settings(path);
                StoredFrame {
                    path: path.clone(),
                    settings,
                    lut_path,
                    preset_name,
                    retouch: self.session.load_retouch(path),
                    crop: self.session.load_crop(path),
                }
            })
            .collect();

        let stored = StoredSession {
            version: SESSION_VERSION,
            frames,
            current_index: self.session.current_index,
            current_path: self.image_path.clone(),
        };

        // No edited frames: do not create an empty sidecar (browsing an untouched
        // folder must not litter `config/sessions/`). Remove a stale one instead,
        // so reverted edits cannot be restored on the next open.
        let result = if stored.frames.is_empty() {
            session_store::remove(&folder)
        } else {
            session_store::save(&folder, &stored)
        };
        if let Err(e) = result {
            eprintln!("⚠ Failed to persist session: {e}");
        }

        // Keep the hero "Continue last session" pointer in step. The config is
        // only rewritten when the folder actually changes; the edit count is
        // refreshed regardless.
        let new_folder = if stored.frames.is_empty() {
            None
        } else {
            Some(folder)
        };
        if self.last_session_folder != new_folder {
            self.last_session_folder = new_folder;
            self.last_session_edits = stored.frames.len();
            self.save_save_settings();
        } else {
            self.last_session_edits = stored.frames.len();
        }
    }

    /// Restores the per-frame state of every currently-open image from the
    /// sidecar of the source folder. Frames that are no longer open are ignored;
    /// the last viewed frame is restored when it is still valid.
    ///
    /// Should be called right after the session's `image_list` is populated —
    /// before the first frame is loaded.
    pub(crate) fn restore_persisted_session(&mut self) {
        let Some(folder) = self.session_folder() else {
            return;
        };
        let Some(stored) = session_store::load(&folder) else {
            return;
        };

        // Canonical lookup of the open frames, so matching does not depend on how
        // the folder was opened (dialog vs drag & drop, relative vs absolute).
        let mut index_of: HashMap<PathBuf, usize> = HashMap::new();
        for (i, p) in self.session.image_list.iter().enumerate() {
            index_of.entry(canonical(p)).or_insert(i);
        }

        let stored_count = stored.frames.len();
        let mut restored = 0usize;
        for frame in &stored.frames {
            let Some(&idx) = index_of.get(&canonical(&frame.path)) else {
                continue;
            };
            // Store under the path the session actually uses (not the stored
            // spelling), so later lookups by `image_list` hit.
            let open_path = self.session.image_list[idx].clone();
            self.session.save_current_settings(
                &open_path,
                frame.settings,
                frame.lut_path.clone(),
                frame.preset_name.clone(),
            );
            self.session.save_retouch(&open_path, frame.retouch.clone());
            self.session.save_crop(&open_path, frame.crop);
            restored += 1;
        }

        if restored > 0 {
            // Prefer the stored current path (robust to a changed folder
            // composition); fall back to the numeric index for older sidecars.
            if let Some(cp) = &stored.current_path {
                if let Some(&idx) = index_of.get(&canonical(cp)) {
                    self.session.current_index = idx;
                }
            } else if stored.current_index < self.session.image_list.len() {
                self.session.current_index = stored.current_index;
            }

            let noun = if restored == 1 { "edit" } else { "edits" };
            if restored < stored_count {
                self.notify(
                    ToastKind::Warning,
                    format!(
                        "Restored {restored} of {stored_count} saved edits — some files are missing"
                    ),
                );
            } else {
                self.notify(
                    ToastKind::Success,
                    format!("Restored {restored} {noun} from the previous session"),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_of_missing_path_is_the_raw_path() {
        // Missing files cannot be canonicalized — the raw path is used as a
        // fallback so matching does not silently drop a frame.
        let p = Path::new("__tiny_luma_no_such_file__.png");
        assert_eq!(canonical(p), p);
    }

    #[test]
    fn load_last_session_ignores_missing_folder() {
        assert_eq!(TinyLumaApp::load_last_session(None), (None, 0));
        assert_eq!(
            TinyLumaApp::load_last_session(Some(String::new())),
            (None, 0)
        );
        assert_eq!(
            TinyLumaApp::load_last_session(Some("__tiny_luma_no_such_dir__".into())),
            (None, 0)
        );
    }
}
