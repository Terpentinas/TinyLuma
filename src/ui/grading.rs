// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! The COLOR GRADING group body: a 2×2 grid of tonal wheels (Shadows,
//! Midtones, Highlights, Global), a Luminance slider for the selected wheel,
//! and the shared Blending / Balance sliders. Drawing and input only — the
//! math lives in `src/pipeline/grading.rs`.
//!
//! Each wheel is drawn from a single cached hue/saturation texture mapped onto
//! a triangle fan, with a draggable puck. Clicking a wheel selects it;
//! dragging sets hue/saturation; double-clicking resets it.

use eframe::egui;
use egui_phosphor::regular as ph;
use std::f32::consts::{FRAC_PI_2, TAU};

use crate::app::TinyLumaApp;
use crate::pipeline::grading::{GradingWheel, hsv_to_rgb};
use crate::theme;
use crate::ui::widgets::labeled_slider;

/// Region captions, in the order of [`ColorGrading::wheels`].
const REGIONS: [&str; 4] = ["Shadows", "Midtones", "Highlights", "Global"];

/// The texture-cache id of the shared hue/saturation wheel.
fn wheel_texture_id() -> egui::Id {
    egui::Id::new("tl_color_grading_wheel")
}

impl TinyLumaApp {
    /// The COLOR GRADING group body.
    /// Returns `(changed, dragging)`.
    pub(crate) fn show_color_grading(&mut self, ui: &mut egui::Ui) -> (bool, bool) {
        let mut changed = false;
        let mut dragging = false;

        // Work on a copy so the wheel closures do not borrow `self` mutably.
        let mut grading = self.settings.grading;
        let mut region = self.grading_region.min(REGIONS.len() - 1);

        let texture = wheel_texture(ui.ctx());
        let gap = 6.0;
        let side = ((ui.available_width() - gap) / 2.0).min(112.0);

        for row in 0..2 {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = gap;
                for col in 0..2 {
                    let i = row * 2 + col;
                    let wheel = match i {
                        0 => &mut grading.shadows,
                        1 => &mut grading.midtones,
                        2 => &mut grading.highlights,
                        _ => &mut grading.global,
                    };
                    let (c, d, clicked) =
                        grading_wheel(ui, &texture, side, REGIONS[i], wheel, region == i);
                    if clicked {
                        region = i;
                    }
                    changed |= c;
                    dragging |= d;
                }
            });
            ui.add_space(2.0);
        }

        // Selected region + reset.
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(
                egui::RichText::new(format!("{} — Luminance", REGIONS[region]))
                    .size(11.0)
                    .color(theme::TEXT_SECONDARY),
            ));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(
                        !grading.is_identity(),
                        egui::Button::new(ph::ARROW_COUNTER_CLOCKWISE).frame(false),
                    )
                    .on_hover_text("Reset all wheels (keeps Blending/Balance)")
                    .clicked()
                {
                    grading.reset_wheels();
                    changed = true;
                }
            });
        });

        // Luminance of the selected wheel.
        let lum = match region {
            0 => &mut grading.shadows.lum,
            1 => &mut grading.midtones.lum,
            2 => &mut grading.highlights.lum,
            _ => &mut grading.global.lum,
        };
        let (c, d) = labeled_slider(ui, "Luminance", lum, -100.0..=100.0, 0.0);
        changed |= c;
        dragging |= d;

        // Shared transition controls.
        ui.add_space(2.0);
        ui.separator();
        ui.add_space(2.0);
        let (c, d) = labeled_slider(ui, "Blending", &mut grading.blending, 0.0..=100.0, 50.0);
        changed |= c;
        dragging |= d;
        let (c, d) = labeled_slider(ui, "Balance", &mut grading.balance, -100.0..=100.0, 0.0);
        changed |= c;
        dragging |= d;

        self.settings.grading = grading;
        self.grading_region = region;
        (changed, dragging)
    }
}

/// Height reserved under each wheel for its caption.
const CAPTION_H: f32 = 14.0;

/// Draws and edits one wheel. Returns `(changed, dragging, clicked)`.
fn grading_wheel(
    ui: &mut egui::Ui,
    texture: &egui::TextureHandle,
    side: f32,
    label: &str,
    wheel: &mut GradingWheel,
    selected: bool,
) -> (bool, bool, bool) {
    let mut changed = false;
    let mut dragging = false;
    let mut clicked = false;

    // The cell is taller than the disc so the caption has its own row.
    let (cell, response) = ui.allocate_exact_size(
        egui::vec2(side, side + CAPTION_H),
        egui::Sense::click_and_drag(),
    );
    let rect = egui::Rect::from_min_size(cell.min, egui::vec2(side, side));
    let center = rect.center();
    let radius = (side * 0.5 - 2.0).max(2.0);

    // --- disc (a textured triangle fan, so the corners stay transparent) ---
    let mut mesh = egui::Mesh::default();
    mesh.texture_id = texture.id();
    mesh.vertices.push(egui::epaint::Vertex {
        pos: center,
        uv: egui::pos2(0.5, 0.5),
        color: egui::Color32::WHITE,
    });
    const SEGMENTS: usize = 64;
    for i in 0..=SEGMENTS {
        let a = i as f32 / SEGMENTS as f32 * TAU;
        let (s, c) = a.sin_cos();
        mesh.vertices.push(egui::epaint::Vertex {
            pos: center + egui::vec2(radius * c, radius * s),
            uv: egui::pos2(0.5 + 0.5 * c, 0.5 + 0.5 * s),
            color: egui::Color32::WHITE,
        });
    }
    for i in 0..SEGMENTS {
        mesh.indices
            .extend_from_slice(&[0, 1 + i as u32, 2 + i as u32]);
    }

    let painter = ui.painter();
    painter.add(egui::Shape::mesh(mesh));
    painter.circle_stroke(
        center,
        radius,
        egui::Stroke::new(
            if selected { 2.0 } else { 1.0 },
            if selected { theme::ACCENT } else { theme::SEPARATOR },
        ),
    );

    // --- input: drag to set hue/saturation, click to select, double-click to reset ---
    if response.double_clicked() {
        *wheel = GradingWheel::default();
        changed = true;
    } else if response.dragged()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let d = pos - center;
        let sat = (d.length() / radius).clamp(0.0, 1.0) * 100.0;
        let hue = d.x.atan2(-d.y).to_degrees().rem_euclid(360.0);
        if (wheel.sat - sat).abs() > 1e-4 || (wheel.hue - hue).abs() > 1e-4 {
            wheel.sat = sat;
            wheel.hue = hue;
            changed = true;
        }
        dragging = true;
        // Dragging a wheel also selects it for the Luminance slider.
        clicked = true;
    } else if response.clicked() {
        clicked = true;
    }

    // --- puck ---
    let sat = (wheel.sat / 100.0).clamp(0.0, 1.0);
    let ang = wheel.hue.to_radians() - FRAC_PI_2;
    let puck = center + egui::vec2(radius * sat * ang.cos(), radius * sat * ang.sin());
    let [r, g, b] = hsv_to_rgb(wheel.hue, sat.max(0.05), 1.0);
    painter.circle_stroke(
        puck,
        5.5,
        egui::Stroke::new(2.5, egui::Color32::from_black_alpha(140)),
    );
    painter.circle_filled(
        puck,
        4.5,
        egui::Color32::from_rgb(
            (r * 255.0 + 0.5) as u8,
            (g * 255.0 + 0.5) as u8,
            (b * 255.0 + 0.5) as u8,
        ),
    );
    painter.circle_stroke(puck, 4.5, egui::Stroke::new(1.0, egui::Color32::WHITE));

    // --- caption under the wheel ---
    painter.text(
        egui::pos2(center.x, rect.bottom() + 2.0),
        egui::Align2::CENTER_TOP,
        label,
        egui::FontId::proportional(10.5),
        if selected {
            theme::TEXT
        } else {
            theme::TEXT_TERTIARY
        },
    );

    response.on_hover_cursor(egui::CursorIcon::PointingHand);

    (changed, dragging, clicked)
}

/// Builds (once per context) and returns the shared hue/saturation wheel
/// texture: hue 0 (red) at the top, increasing clockwise; saturation grows
/// outward.
fn wheel_texture(ctx: &egui::Context) -> egui::TextureHandle {
    let id = wheel_texture_id();
    if let Some(tex) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(id)) {
        return tex;
    }

    const N: usize = 128;
    let mut rgb = vec![0u8; N * N * 3];
    for y in 0..N {
        for x in 0..N {
            let dx = (x as f32 + 0.5) / N as f32 * 2.0 - 1.0;
            let dy = (y as f32 + 0.5) / N as f32 * 2.0 - 1.0;
            let hue = dx.atan2(-dy).to_degrees().rem_euclid(360.0);
            let sat = (dx * dx + dy * dy).sqrt().min(1.0);
            let [r, g, b] = hsv_to_rgb(hue, sat, 1.0);
            let idx = (y * N + x) * 3;
            rgb[idx] = (r * 255.0 + 0.5) as u8;
            rgb[idx + 1] = (g * 255.0 + 0.5) as u8;
            rgb[idx + 2] = (b * 255.0 + 0.5) as u8;
        }
    }

    let image = egui::ColorImage::from_rgb([N, N], &rgb);
    let tex = ctx.load_texture("tl_color_grading_wheel", image, egui::TextureOptions::LINEAR);
    ctx.data_mut(|d| d.insert_temp(id, tex.clone()));
    tex
}
