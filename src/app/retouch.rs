// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! Glue between the pure retouch layer (`src/retouch/`) and the app state,
//! caches and pipeline. No pixel math lives here.

use eframe::egui;
use image::RgbImage;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::heal_job::{HealReply, HealRequest, HealWorker};
use super::TinyLumaApp;
use crate::history::Snapshot;
use crate::retouch::raster::Rect;
use crate::retouch::spot::{apply_spot_cached, SpotCache};
use crate::retouch::{BrushSettings, RetouchLayer, Spot};

/// Longest side of the full-resolution working image used while the retouch tool
/// is active. Big enough to see real pixels, small enough to stay responsive.
const RETOUCH_MAX_DIM: u32 = 4096;

/// Retouch tool + layer + caches, grouped so `TinyLumaApp` stays readable.
pub(crate) struct RetouchState {
    /// The edit itself (undoable).
    pub(crate) layer: RetouchLayer,
    /// Tool is active (canvas shows the retouch cursor, options bar appears).
    pub(crate) active: bool,
    /// Current brush.
    pub(crate) brush: BrushSettings,
    /// Cached healed preview base + the layer revision it was built from.
    pub(crate) healed_cache: Option<(u64, RgbImage)>,
    /// Bumped on every layer change; the heal cache is valid only when equal.
    pub(crate) revision: u64,
    /// Memoized healed regions, so rebuilding the layer (undo/redo, frame
    /// switch, export) does not re-run the expensive content-aware fills.
    pub(crate) spot_cache: SpotCache,
    /// A pointer gesture is in progress (one history entry per gesture).
    pub(crate) stroke_active: bool,
    /// State before the current gesture (pushed to history on release).
    pub(crate) stroke_baseline: Option<Snapshot>,
    /// Painted path of the current gesture, normalized `0..1`. Nothing is healed
    /// until the pointer is released (Photopea-style live indication).
    pub(crate) pending_path: Vec<[f32; 2]>,
    /// Cached indication texture for the pending path + the image area it covers.
    pub(crate) overlay_tex: Option<egui::TextureHandle>,
    pub(crate) overlay_area: Option<[f32; 4]>,
    /// The pending path changed — the indication texture must be rebuilt.
    pub(crate) overlay_dirty: bool,
    /// Full-resolution original (long side ≤ 4096), loaded while the tool is
    /// active. Shared with the worker by `Arc`, so enqueuing a rebuild is a
    /// pointer clone rather than a tens-of-MB copy.
    pub(crate) full_base: Option<Arc<RgbImage>>,
    /// Full-resolution healed image, rebuilt from `full_base` when needed.
    pub(crate) full_healed: Option<Arc<RgbImage>>,
    /// Source path of `full_base`, to reload it on a frame switch.
    pub(crate) full_path: Option<PathBuf>,
    /// The full-resolution textures must be (re)uploaded.
    pub(crate) full_texture_dirty: bool,
    /// The healed image is out of date and a rebuild is needed.
    pub(crate) full_healed_stale: bool,
    /// Region of `full_healed` that changed and still needs a partial
    /// upload (consumed by `upload_full_retouch_textures`).
    pub(crate) full_dirty_rect: Option<Rect>,
    /// Background heal worker (decode + layer rebuild). Lazily spawned.
    pub(crate) worker: Option<HealWorker>,
    /// Monotonically increasing job revision (coalescing key).
    pub(crate) job_revision: u64,
    /// A full-resolution decode is in flight.
    pub(crate) full_loading: bool,
    /// A healed full-res rebuild is queued or running ("Healing…").
    pub(crate) render_pending: bool,
    /// Revision of the latest rebuild we care about.
    pub(crate) pending_job_rev: Option<u64>,
    /// The heal layer when the tool was opened (or after the last undo/redo
    /// while it was open). Cancelling the tool (`Esc`) restores it.
    pub(crate) session_base: Option<RetouchLayer>,
    /// `history.len()` when the session started; `Esc` truncates back to it.
    pub(crate) session_hist_len: usize,
}

impl Default for RetouchState {
    fn default() -> Self {
        Self {
            layer: RetouchLayer::default(),
            active: false,
            brush: BrushSettings::default(),
            healed_cache: None,
            revision: 0,
            spot_cache: SpotCache::default(),
            stroke_active: false,
            stroke_baseline: None,
            pending_path: Vec::new(),
            overlay_tex: None,
            overlay_area: None,
            overlay_dirty: false,
            full_base: None,
            full_healed: None,
            full_path: None,
            full_texture_dirty: false,
            full_healed_stale: false,
            full_dirty_rect: None,
            worker: None,
            job_revision: 0,
            full_loading: false,
            render_pending: false,
            pending_job_rev: None,
            session_base: None,
            session_hist_len: 0,
        }
    }
}

impl RetouchState {
    /// Applies `spots` in order, consulting the memoization cache. Replaying an
    /// unchanged prefix (undo/redo, rebuild) hits the cache and skips the fills.
    fn apply_spots_cached(cache: &mut SpotCache, spots: &[Spot], img: &mut RgbImage) {
        for spot in spots {
            apply_spot_cached(cache, img, spot);
        }
    }

    /// Rebuilds the healed base if the cache is stale for the current layer and
    /// image size.
    pub(crate) fn ensure_healed(&mut self, base: &RgbImage) {
        if self.layer.is_empty() {
            self.healed_cache = None;
            return;
        }
        let dims = (base.width(), base.height());
        if let Some((rev, img)) = &self.healed_cache {
            if *rev == self.revision && (img.width(), img.height()) == dims {
                return;
            }
        }
        let mut img = base.clone();
        Self::apply_spots_cached(&mut self.spot_cache, &self.layer.spots, &mut img);
        self.healed_cache = Some((self.revision, img));
    }

    /// The current healed base, rebuilding it if needed.
    pub(crate) fn healed_base<'a>(&'a mut self, base: &'a RgbImage) -> &'a RgbImage {
        self.ensure_healed(base);
        match &self.healed_cache {
            Some((_, img)) => img,
            None => base,
        }
    }

    /// Replaces the layer (frame switch, undo/redo) and invalidates the cache.
    pub(crate) fn set_layer(&mut self, layer: RetouchLayer) {
        self.layer = layer;
        self.revision = self.revision.wrapping_add(1);
        self.healed_cache = None;
        // A different layer invalidates every pending full-res job.
        if let Some(w) = &mut self.worker {
            w.invalidate();
        }
        self.job_revision = self.job_revision.wrapping_add(1);
        self.full_healed = None;
        self.full_healed_stale = true;
        self.full_texture_dirty = true;
        self.full_dirty_rect = None;
        self.render_pending = false;
        self.pending_job_rev = None;
        self.stroke_active = false;
        self.stroke_baseline = None;
        self.pending_path.clear();
        self.overlay_tex = None;
        self.overlay_area = None;
        self.overlay_dirty = false;
    }

    /// Frees the full-resolution buffers (on tool deactivation) and cancels any
    /// pending decode/rebuild.
    pub(crate) fn unload_full(&mut self) {
        if let Some(w) = &mut self.worker {
            w.invalidate();
        }
        self.full_base = None;
        self.full_healed = None;
        self.full_path = None;
        self.full_texture_dirty = false;
        self.full_healed_stale = false;
        self.full_dirty_rect = None;
        self.full_loading = false;
        self.render_pending = false;
        self.pending_job_rev = None;
    }

    // The full-resolution decode and whole-layer rebuild now run on the
    // background heal worker (`super::heal_job`). See `request_full_base`,
    // `ensure_full_rebuild` and `poll_heal` in the `TinyLumaApp` impl below:
    // nothing here touches the UI thread with a multi-second fill any more.

    /// Adds a spot, updating the heal cache incrementally when possible.
    pub(crate) fn add_spot(&mut self, spot: Spot, base: &RgbImage) {
        self.layer.push(spot.clone());
        let dims = (base.width(), base.height());
        let incremental = matches!(
            &self.healed_cache,
            Some((rev, img)) if *rev == self.revision && (img.width(), img.height()) == dims
        );
        if incremental {
            // Take the image out so the cache and the image can be borrowed at
            // the same time (they are disjoint fields, but a method call cannot split them).
            let mut img = self.healed_cache.take().unwrap().1;
            apply_spot_cached(&mut self.spot_cache, &mut img, &spot);
            self.revision = self.revision.wrapping_add(1);
            self.healed_cache = Some((self.revision, img));
            return;
        }
        self.revision = self.revision.wrapping_add(1);
        self.ensure_healed(base);
    }
}

impl TinyLumaApp {
    /// Records the pre-gesture state and starts a stroke. Nothing is healed yet:
    /// the painted region is previewed as a translucent overlay and only applied
    /// when the pointer is released.
    pub(crate) fn retouch_begin_stroke(&mut self, pos: [f32; 2]) {
        self.retouch.stroke_active = true;
        self.retouch.pending_path.clear();
        self.retouch.pending_path.push(pos);
        self.retouch.overlay_dirty = true;
        self.retouch.stroke_baseline = Some(self.snapshot());
    }

    /// Extends the pending stroke with a pointer position (normalized).
    pub(crate) fn retouch_extend_stroke(&mut self, pos: [f32; 2]) {
        if self
            .retouch
            .pending_path
            .last()
            .map_or(true, |last| *last != pos)
        {
            self.retouch.pending_path.push(pos);
            self.retouch.overlay_dirty = true;
        }
    }

    /// Ends a gesture: heals the whole painted path and records one history
    /// entry for it.
    pub(crate) fn retouch_commit_stroke(&mut self) {
        self.retouch.stroke_active = false;
        let baseline = self.retouch.stroke_baseline.take();
        let path = std::mem::take(&mut self.retouch.pending_path);

        if !path.is_empty() {
            let (ww, wh) = self.retouch_work_dims();
            // One painted stroke becomes one op, healed by a single union fill
            // (`apply_stroke`) instead of a fill per dab.
            let stroke = self.retouch.brush.to_stroke(&path, ww, wh);
            if let Some(base) = self.preview_base.as_ref() {
                // Preview cache (used when the tool is off / exported preview).
                self.retouch.add_spot(stroke.clone(), base);
            }
            // Full-resolution working image: the layer is the source of truth.
            // Rebuilds it on the background worker — here we only flag it
            // stale and let `ensure_full_rebuild` enqueue the job on a later
            // frame, so the pointer release never blocks on the fill.
            if self.retouch.active {
                self.retouch.full_healed_stale = true;
                self.retouch.render_pending = false;
            }
        }

        if let Some(base_snap) = baseline {
            if base_snap.retouch != self.retouch.layer {
                self.history.push(base_snap);
            }
        }
        if let Some(path) = self.image_path.clone() {
            let layer = self.retouch.layer.clone();
            self.session.save_retouch(&path, layer);
        }
        self.mark_retouch_dirty();
    }

    /// Cancels an in-progress gesture: the painted region is discarded, nothing
    /// was applied to the layer yet.
    pub(crate) fn retouch_cancel_stroke(&mut self) {
        self.retouch.stroke_active = false;
        self.retouch.pending_path.clear();
        self.retouch.stroke_baseline = None;
    }

    /// Leaves the retouch tool and throws away every heal made while it was
    /// open (`Esc`), including their history entries so redo cannot bring them
    /// back. An in-progress gesture is discarded first.
    pub(crate) fn retouch_cancel(&mut self) {
        if self.retouch.stroke_active {
            self.retouch_cancel_stroke();
        }
        if let Some(base) = self.retouch.session_base.take() {
            self.retouch.set_layer(base);
            self.history.truncate(self.retouch.session_hist_len);
            if let Some(path) = self.image_path.clone() {
                let layer = self.retouch.layer.clone();
                self.session.save_retouch(&path, layer);
            }
            self.mark_retouch_dirty();
        }
        // Closes the tool and clears the session fields.
        self.retouch_toggle();
    }

    /// Clears every heal edit on the current image as a single undo step,
    /// mirroring the crop tool's `Reset`. The brush size/hardness are settings
    /// and are left untouched.
    pub(crate) fn retouch_reset(&mut self) {
        // A mid-air gesture is discarded, then nothing else would be cleared.
        if self.retouch.stroke_active {
            self.retouch_cancel_stroke();
        }
        if self.retouch.layer.is_empty() {
            return;
        }
        // Record the pre-clear layer so one Ctrl+Z brings all the heals back.
        self.history.push(self.snapshot());
        self.retouch.set_layer(RetouchLayer::default());
        if let Some(path) = self.image_path.clone() {
            let layer = self.retouch.layer.clone();
            self.session.save_retouch(&path, layer);
        }
        self.mark_retouch_dirty();
    }

    /// Toggles the retouch tool and forces a re-render.
    pub(crate) fn retouch_toggle(&mut self) {
        // Never leave a half-finished gesture behind.
        if self.retouch.stroke_active {
            self.retouch_commit_stroke();
        }
        // Crop and retouch both own the canvas: leaving retouch is implicit.
        self.crop.active = false;
        self.wb_picker_active = false;
        self.retouch.active = !self.retouch.active;
        // A session runs from activation (or the last undo/redo while active)
        // to deactivation; `Esc` reverts to `session_base`.
        if self.retouch.active {
            self.retouch.session_base = Some(self.retouch.layer.clone());
            self.retouch.session_hist_len = self.history.len();
            // The retouch tool owns the canvas; a proxy render in
            // flight is no longer wanted (its reply would be discarded anyway).
            self.invalidate_proxy_render();
        } else {
            self.retouch.session_base = None;
            self.retouch.session_hist_len = 0;
        }
        self.retouch.stroke_active = false;
        self.retouch.pending_path.clear();
        self.retouch.stroke_baseline = None;
        if !self.retouch.active {
            // Leaving the tool: make sure no worker is mid-fill on the buffers
            // we are about to free, then free them.
            self.drain_full_heal();
            self.retouch.unload_full();
        }
        // Entering: keep the current zoom and pan. `zoom_scale` is relative to
        // "fit" and `pan_offset` is in screen pixels, so the framing is
        // preserved when the canvas switches from the preview to the
        // full-resolution working image. Resetting to fit here would throw away
        // the user's zoom right where they spotted the defect.
        self.color_dirty = true;
        self.spatial_dirty = true;
        self.full_render_pending = true;
        self.mark_retouch_dirty();
        // Persist the brush now (not only on exit): toggling is infrequent, and a
        // crash would otherwise lose the user's comfortable size/hardness.
        self.save_save_settings();
    }

    /// Flags a retouch change: re-run the heal + pipeline and drop the stale
    /// cached render of this frame.
    pub(crate) fn mark_retouch_dirty(&mut self) {
        self.retouch_dirty = true;
        self.color_dirty = true;
        self.spatial_dirty = true;
        self.mark_session_dirty();
        if let Some(path) = &self.image_path {
            self.preview_cache.drop_render(path);
        }
    }

    /// Pixel dimensions the brush operates in: the full-resolution working image
    /// while the tool is active, otherwise the preview.
    pub(crate) fn retouch_work_dims(&self) -> (usize, usize) {
        if self.retouch.active {
            if let Some(full) = &self.retouch.full_base {
                return (full.width() as usize, full.height() as usize);
            }
        }
        self.preview_base
            .as_ref()
            .map_or((1, 1), |b| (b.width() as usize, b.height() as usize))
    }

    /// (Re)uploads the full-resolution retouch textures: the healed image into
    /// `self.texture` (shown on the canvas) and the original into
    /// `self.original_texture` (before/after split).
    ///
    /// A decode or a size change does a full upload; a healed result
    /// that changed only a region is uploaded through
    /// [`egui::TextureHandle::set_partial`] (verified available in epaint 0.28).
    pub(crate) fn upload_full_retouch_textures(&mut self, ctx: &egui::Context) {
        let Some(healed) = self.retouch.full_healed.as_ref() else {
            return;
        };
        let (w, h) = (healed.width() as usize, healed.height() as usize);

        if self.retouch.full_texture_dirty {
            let healed_img = egui::ColorImage::from_rgb([w, h], healed.as_raw());
            let original_img = self
                .retouch
                .full_base
                .as_ref()
                .map(|base| egui::ColorImage::from_rgb([w, h], base.as_raw()));

            if let Some(tex) = &mut self.texture {
                tex.set(healed_img, egui::TextureOptions::LINEAR);
            } else {
                self.texture = Some(ctx.load_texture(
                    "retouch_full",
                    healed_img,
                    egui::TextureOptions::LINEAR,
                ));
            }

            if let Some(original_img) = original_img {
                if let Some(tex) = &mut self.original_texture {
                    tex.set(original_img, egui::TextureOptions::LINEAR);
                } else {
                    self.original_texture = Some(ctx.load_texture(
                        "retouch_full_orig",
                        original_img,
                        egui::TextureOptions::LINEAR,
                    ));
                }
            }

            self.retouch.full_texture_dirty = false;
            self.retouch.full_dirty_rect = None;
            // These full-res textures are not a valid preview upload: invalidate
            // the upload keys so returning to the preview re-uploads.
            self.upload_display_key = None;
            self.upload_orig_key = None;
            return;
        }

        // Region-only update of the healed texture (the original never changes
        // on a heal). Falls back to a full upload if the texture is missing.
        if let Some(rect) = self.retouch.full_dirty_rect.take() {
            // `set_partial` debug-asserts that the region fits the texture, so
            // clamp defensively before uploading.
            if let Some(rect) = clamp_rect(rect, w, h) {
                match &mut self.texture {
                    Some(tex) => {
                        let sub = region_color_image(healed, rect);
                        tex.set_partial([rect.x, rect.y], sub, egui::TextureOptions::LINEAR);
                    }
                    None => self.retouch.full_texture_dirty = true,
                }
            }
        }
    }

    /// Lazily spawns the background heal worker.
    fn heal_worker(&mut self, ctx: &egui::Context) -> &mut HealWorker {
        if self.retouch.worker.is_none() {
            self.retouch.worker = Some(HealWorker::spawn(ctx));
        }
        self.retouch
            .worker
            .as_mut()
            .expect("worker spawned just above")
    }

    /// Requests the full-resolution base decode for `path`, off the UI thread,
    /// unless it is already loaded or already loading for that path.
    pub(crate) fn request_full_base(&mut self, ctx: &egui::Context, path: Option<&Path>) {
        let Some(path) = path else {
            return;
        };
        let same = self.retouch.full_path.as_deref() == Some(path);
        if same && (self.retouch.full_base.is_some() || self.retouch.full_loading) {
            return;
        }
        self.retouch.job_revision = self.retouch.job_revision.wrapping_add(1);
        let revision = self.retouch.job_revision;
        // Drop any previous full-res state; the decode reply fills it in.
        self.retouch.full_base = None;
        self.retouch.full_healed = None;
        self.retouch.full_healed_stale = true;
        self.retouch.full_dirty_rect = None;
        self.retouch.render_pending = false;
        self.retouch.pending_job_rev = None;
        self.retouch.full_path = Some(path.to_path_buf());
        self.retouch.full_loading = true;
        let req = HealRequest::Decode {
            revision,
            path: path.to_path_buf(),
            max_dim: RETOUCH_MAX_DIM,
        };
        self.heal_worker(ctx).enqueue(req);
    }

    /// Enqueues a full-res heal rebuild when the healed layer is stale and none
    /// is already queued. Cheap when nothing changed (returns early).
    pub(crate) fn ensure_full_rebuild(&mut self, ctx: &egui::Context) {
        if !self.retouch.active || !self.retouch.full_healed_stale || self.retouch.render_pending {
            return;
        }
        let Some(base) = self.retouch.full_base.as_ref() else {
            return;
        };
        // `Arc` share: the worker clones the pixels itself, off this thread.
        let base = Arc::clone(base);
        let spots = self.retouch.layer.spots.clone();
        self.retouch.job_revision = self.retouch.job_revision.wrapping_add(1);
        let revision = self.retouch.job_revision;
        self.retouch.render_pending = true;
        self.retouch.pending_job_rev = Some(revision);
        let req = HealRequest::Rebuild {
            revision,
            base,
            spots,
        };
        self.heal_worker(ctx).enqueue(req);
    }

    /// Applies finished heal jobs. Called once per frame before the pipeline;
    /// never blocks. Stale replies are discarded by revision.
    pub(crate) fn poll_heal(&mut self) {
        // Retry a coalesced job if the worker became free.
        if let Some(w) = &mut self.retouch.worker {
            w.pump();
        }
        let mut replies = Vec::new();
        if let Some(w) = &self.retouch.worker {
            while let Ok(reply) = w.rx.try_recv() {
                replies.push(reply);
            }
        }
        for reply in replies {
            self.apply_heal_reply(reply);
        }
    }

    /// Applies one worker reply, discarding anything superseded.
    fn apply_heal_reply(&mut self, reply: HealReply) {
        match reply {
            HealReply::Decoded {
                revision,
                path,
                image,
            } => {
                if revision != self.retouch.job_revision
                    || Some(&path) != self.image_path.as_ref()
                {
                    return;
                }
                let base = Arc::new(image);
                self.retouch.full_path = Some(path);
                self.retouch.full_loading = false;
                self.retouch.full_healed_stale = !self.retouch.layer.is_empty();
                // Show the decoded base immediately; the rebuild overwrites it.
                self.retouch.full_healed = Some(Arc::clone(&base));
                self.retouch.full_base = Some(base);
                self.retouch.full_texture_dirty = true;
                self.retouch.full_dirty_rect = None;
                // Force the retouch branch of `process_preview` to run + upload.
                self.full_render_pending = true;
            }
            HealReply::Healed {
                revision,
                image,
                dirty,
            } => {
                if revision != self.retouch.job_revision || !self.retouch.active {
                    return;
                }
                self.retouch.render_pending = false;
                self.retouch.pending_job_rev = None;
                self.retouch.full_healed = Some(Arc::new(image));
                self.retouch.full_healed_stale = false;
                self.retouch.full_dirty_rect = dirty;
                if dirty.is_none() {
                    // The layer is empty: the healed image equals the base.
                    self.retouch.full_texture_dirty = true;
                }
                self.full_render_pending = true;
            }
            HealReply::Failed { revision } => {
                if revision == self.retouch.job_revision {
                    self.retouch.full_loading = false;
                    self.retouch.full_healed_stale = false;
                    self.full_render_pending = true;
                }
            }
        }
    }

    /// Blocks until the pending full-res rebuild is applied. Used before export
    /// and when leaving the tool so the full-res layer is consistent. A 2 s
    /// timeout on each wait guarantees a dropped worker cannot hang the app.
    pub(crate) fn drain_full_heal(&mut self) {
        if self.retouch.worker.is_none() {
            return;
        }
        if let Some(w) = &mut self.retouch.worker {
            w.pump();
        }
        // Drain everything already ready.
        loop {
            let reply = match &self.retouch.worker {
                Some(w) => w.rx.try_recv(),
                None => break,
            };
            match reply {
                Ok(r) => self.apply_heal_reply(r),
                Err(_) => break,
            }
        }
        // Wait for the pending rebuild, if any.
        while self.retouch.render_pending {
            let reply = match &self.retouch.worker {
                Some(w) => w.rx.recv_timeout(std::time::Duration::from_millis(2000)),
                None => break,
            };
            match reply {
                Ok(r) => self.apply_heal_reply(r),
                Err(_) => break,
            }
        }
    }
}

/// Clamps `rect` to an image of `w`×`h`, returning `None` if it becomes empty.
fn clamp_rect(rect: Rect, w: usize, h: usize) -> Option<Rect> {
    let x0 = rect.x.min(w);
    let y0 = rect.y.min(h);
    let x1 = (rect.x + rect.w).min(w);
    let y1 = (rect.y + rect.h).min(h);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(Rect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    })
}

/// Copies the `rect` sub-region of `img` into a fresh `ColorImage` for a
/// partial texture update (`TextureHandle::set_partial`).
fn region_color_image(img: &RgbImage, rect: Rect) -> egui::ColorImage {
    let (w, h) = (rect.w, rect.h);
    let img_w = img.width() as usize;
    let raw = img.as_raw();
    let mut bytes = Vec::with_capacity(w * h * 3);
    for y in rect.y..rect.y + h {
        let start = (y * img_w + rect.x) * 3;
        bytes.extend_from_slice(&raw[start..start + w * 3]);
    }
    egui::ColorImage::from_rgb([w, h], &bytes)
}
