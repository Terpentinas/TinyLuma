// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! White-balance picker (eyedropper). Click a neutral area of the canvas to set
//! Temp/Tint so that area renders neutral.
//!
//! The colour math lives in `src/pipeline/color.rs` (`WhiteBalance`,
//! `sliders_from_linear`); this module handles the tool state and the screen →
//! source mapping (the inverse of `crop_tool::draw_transformed`).

use eframe::egui;

use crate::app::TinyLumaApp;
use crate::color::oklab::srgb_u8_linear_table;
use crate::crop::{NormRect, Orientation};
use crate::pipeline::color::sliders_from_linear;

/// Half-size of the averaged sampling window (source pixels).
const SAMPLE_RADIUS: isize = 2;

impl TinyLumaApp {
    /// Toggles the white-balance picker. Crop and retouch also own the canvas
    /// pointer, so enabling the picker disables them.
    pub(crate) fn wb_picker_toggle(&mut self) {
        if self.wb_picker_active {
            self.wb_picker_active = false;
            return;
        }
        if self.crop.active {
            self.crop_toggle();
        }
        if self.retouch.active {
            self.retouch_toggle();
        }
        self.wb_picker_active = true;
    }

    /// Handles the picker's canvas input. Returns `true` when the picker owns
    /// the left click this frame (so the canvas must not zoom).
    pub(crate) fn wb_picker_pointer(
        &mut self,
        ui: &mut egui::Ui,
        image_rect: egui::Rect,
        hover_pos: Option<egui::Pos2>,
        visible: NormRect,
        orientation: Orientation,
        angle: f32,
    ) -> bool {
        if !self.wb_picker_active || image_rect.width() <= 0.0 || image_rect.height() <= 0.0 {
            return false;
        }
        let over = hover_pos.is_some_and(|p| image_rect.contains(p));
        let space_held = ui.input(|i| i.key_down(egui::Key::Space));
        if ui.input(|i| i.pointer.primary_clicked()) && over && !space_held {
            if let Some(p) = hover_pos {
                self.wb_pick_at(p, image_rect, visible, orientation, angle);
            }
            return true;
        }
        // Own the canvas (block the zoom-click) but still allow Space+drag panning.
        over && !space_held
    }

    /// Samples `preview_base` under `pos` and sets Temp/Tint to neutralize it.
    fn wb_pick_at(
        &mut self,
        pos: egui::Pos2,
        image_rect: egui::Rect,
        visible: NormRect,
        orientation: Orientation,
        angle: f32,
    ) {
        let Some(base) = self.preview_base.as_ref() else {
            return;
        };
        let (vw, vh) = (base.width() as usize, base.height() as usize);
        if vw == 0 || vh == 0 {
            return;
        }

        // screen -> visible-space normalized (inverse of `draw_transformed`)
        let rx = visible.x + (pos.x - image_rect.left()) / image_rect.width() * visible.w;
        let ry = visible.y + (pos.y - image_rect.top()) / image_rect.height() * visible.h;
        // Undo the straighten rotation, then map the oriented coords to texture UV.
        let aspect = self.crop_image_ratio();
        let (x, y) = rotate_inverse(rx, ry, angle, aspect);
        let (u, v) = orientation.to_texture(x, y);
        let cx = (u * vw as f32).floor().clamp(0.0, (vw - 1) as f32) as isize;
        let cy = (v * vh as f32).floor().clamp(0.0, (vh - 1) as f32) as isize;

        let raw = base.as_raw();
        let lin = srgb_u8_linear_table();
        let mut sum = [0.0f64; 3];
        let mut n = 0.0f64;
        for dy in -SAMPLE_RADIUS..=SAMPLE_RADIUS {
            for dx in -SAMPLE_RADIUS..=SAMPLE_RADIUS {
                let sx = cx + dx;
                let sy = cy + dy;
                if sx < 0 || sy < 0 || sx >= vw as isize || sy >= vh as isize {
                    continue;
                }
                let i = (sy as usize * vw + sx as usize) * 3;
                // Skip blown-out pixels: they carry no colour information.
                if raw[i] >= 250 && raw[i + 1] >= 250 && raw[i + 2] >= 250 {
                    continue;
                }
                sum[0] += lin[raw[i] as usize] as f64;
                sum[1] += lin[raw[i + 1] as usize] as f64;
                sum[2] += lin[raw[i + 2] as usize] as f64;
                n += 1.0;
            }
        }
        if n < 1.0 {
            return;
        }
        let sample = [(sum[0] / n) as f32, (sum[1] / n) as f32, (sum[2] / n) as f32];
        let Some((temp, tint)) = sliders_from_linear(sample) else {
            return;
        };

        if (self.settings.temp - temp).abs() < 0.01 && (self.settings.tint - tint).abs() < 0.01 {
            return;
        }
        // One undoable change. The center panel runs after the frame's history
        // block, so we record it explicitly.
        self.history.push(self.snapshot());
        self.settings.temp = temp;
        self.settings.tint = tint;
        self.color_dirty = true;
        self.full_render_pending = true;
        if let Some(path) = self.image_path.clone() {
            let preset_name = self.current_preset_name();
            self.session
                .save_current_settings(&path, self.settings, self.lut_path.clone(), preset_name);
        }
        self.mark_session_dirty();
        self.wb_picker_active = false;
    }
}

/// Inverse of `crop::orient::rotate_point`, which `draw_transformed` applies
/// before mapping the visible sub-rect to the screen.
fn rotate_inverse(x: f32, y: f32, deg: f32, aspect: f32) -> (f32, f32) {
    let aspect = if aspect > 0.0 { aspect } else { 1.0 };
    let (s, c) = deg.to_radians().sin_cos();
    let dx = x - 0.5;
    let dy = y - 0.5;
    (0.5 + c * dx + s * dy / aspect, 0.5 - s * dx * aspect + c * dy)
}
