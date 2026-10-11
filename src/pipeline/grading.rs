// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! Color Grading / Split Toning in Oklab.
//!
//! Four tonal wheels — Shadows, Midtones, Highlights and Global — each add a
//! hue/saturation tint plus a lightness shift, weighted by how close a pixel's
//! Oklab lightness is to that region. `Blending` softens the zone transitions
//! and `Balance` slides the shadow/highlight midpoint.
//!
//! A wheel is only a constant `(a, b)` offset and an `L` shift in Oklab, so the
//! whole effect is per-pixel and bakes into the combined 3D LUT for free. The
//! layout follows Lightroom's Color Grading panel; the Oklab formulation is
//! adapted from lightcraft (`crates/pipeline/src/colorops.rs`, MIT/Apache-2.0).

use serde::{Deserialize, Serialize};

use crate::color::oklab::{rgb_to_oklab, srgb_to_linear};

/// Maximum `(a, b)` magnitude added at saturation 100.
///
/// Fitted numerically to match lightcraft's `ColorOps::grade` by the
/// calibration kit (`cargo test calibration -- --ignored`); 0.09 makes the
/// mean ΔE zero over the colour chart.
pub(crate) const SAT_SCALE: f32 = 0.09;
/// Maximum `L` shift added at luminance 100.
pub(crate) const LUM_SCALE: f32 = 0.12;
/// Midpoint of the shadow/highlight transition at Balance 0.
const BASE_MIDPOINT: f32 = 0.5;
/// How far Balance (±1) may move that midpoint.
const BALANCE_HALF: f32 = 0.25;
/// Base half-width of the transition; widened by Blending.
const BASE_WIDTH: f32 = 0.15;
/// How much Blending (0..1) widens the transition.
const BLEND_WIDTH: f32 = 0.5;

/// One tonal wheel: a hue angle (degrees), saturation and luminance, both the
/// latter on a −100..100 scale.
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize, Debug, Default)]
#[serde(default)]
pub(crate) struct GradingWheel {
    /// Hue in degrees, `0..=360` (the sRGB/HSV hue shown on the wheel).
    pub(crate) hue: f32,
    /// Saturation, `0..=100`.
    pub(crate) sat: f32,
    /// Luminance shift, `-100..=100`.
    pub(crate) lum: f32,
}

impl GradingWheel {
    /// Whether this wheel changes nothing.
    pub(crate) fn is_identity(&self) -> bool {
        self.sat.abs() < 1e-6 && self.lum.abs() < 1e-6
    }
}

/// The COLOR GRADING settings. `serde(default)` keeps older presets/configs
/// loading as a no-op.
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize, Debug)]
#[serde(default)]
pub(crate) struct ColorGrading {
    pub(crate) shadows: GradingWheel,
    pub(crate) midtones: GradingWheel,
    pub(crate) highlights: GradingWheel,
    pub(crate) global: GradingWheel,
    /// Softness of the zone transitions, `0..=100` (default 50).
    pub(crate) blending: f32,
    /// Shifts the shadow/highlight midpoint, `-100..=100` (default 0).
    pub(crate) balance: f32,
}

impl Default for ColorGrading {
    fn default() -> Self {
        Self {
            shadows: GradingWheel::default(),
            midtones: GradingWheel::default(),
            highlights: GradingWheel::default(),
            global: GradingWheel::default(),
            blending: 50.0,
            balance: 0.0,
        }
    }
}

impl ColorGrading {
    /// The four wheels in the order Shadows, Midtones, Highlights, Global.
    pub(crate) fn wheels(&self) -> [&GradingWheel; 4] {
        [&self.shadows, &self.midtones, &self.highlights, &self.global]
    }

    /// Wheels and shifts to `GradingWheel::default()` while keeping Blending
    /// and Balance.
    pub(crate) fn reset_wheels(&mut self) {
        self.shadows = GradingWheel::default();
        self.midtones = GradingWheel::default();
        self.highlights = GradingWheel::default();
        self.global = GradingWheel::default();
    }

    /// Whether grading is a no-op.
    pub(crate) fn is_identity(&self) -> bool {
        self.wheels().iter().all(|w| w.is_identity())
    }
}

/// A wheel resolved for the pixel loop: the Oklab chroma offset `(a, b)` and
/// the `L` shift. Precomputed once per render so `cos`/`sin` and the sRGB→Oklab
/// hue conversion stay out of the per-pixel path.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct GradingParams {
    /// `[a, b, L]` per region (Shadows, Midtones, Highlights, Global).
    wheels: [[f32; 3]; 4],
    /// Blending, `0..=1`.
    blending: f32,
    /// Balance, `-1..=1`.
    balance: f32,
}

impl GradingParams {
    pub(crate) fn new(g: &ColorGrading) -> Self {
        Self::with_scales(g, SAT_SCALE, LUM_SCALE)
    }

    /// Like [`new`](Self::new), but with explicit scale constants. Used by the
    /// calibration kit to fit `SAT_SCALE`/`LUM_SCALE` against a reference.
    pub(crate) fn with_scales(g: &ColorGrading, sat_scale: f32, lum_scale: f32) -> Self {
        Self {
            wheels: [
                wheel_offset(&g.shadows, sat_scale, lum_scale),
                wheel_offset(&g.midtones, sat_scale, lum_scale),
                wheel_offset(&g.highlights, sat_scale, lum_scale),
                wheel_offset(&g.global, sat_scale, lum_scale),
            ],
            blending: (g.blending / 100.0).clamp(0.0, 1.0),
            balance: (g.balance / 100.0).clamp(-1.0, 1.0),
        }
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.wheels
            .iter()
            .all(|w| w[0].abs() < 1e-6 && w[1].abs() < 1e-6 && w[2].abs() < 1e-6)
    }
}

/// Resolves one wheel into an `[a, b, L]` Oklab offset.
fn wheel_offset(w: &GradingWheel, sat_scale: f32, lum_scale: f32) -> [f32; 3] {
    let sat = (w.sat / 100.0).max(0.0) * sat_scale;
    let hue = srgb_hue_to_oklab_hue(w.hue);
    [
        sat * hue.cos(),
        sat * hue.sin(),
        (w.lum / 100.0) * lum_scale,
    ]
}

#[inline]
fn smoothstep(x0: f32, x1: f32, x: f32) -> f32 {
    let t = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Applies color grading to an Oklab pixel, tinting each tonal zone by its
/// wheel. A no-op when all the wheels are neutral.
#[inline]
pub(crate) fn apply_grading(p: &mut [f32; 3], g: &GradingParams) {
    if g.is_identity() {
        return;
    }
    let l = p[0];
    // `balance > 0` lowers the midpoint (more shadow influence).
    let midpoint = BASE_MIDPOINT - g.balance * BALANCE_HALF;
    let width = BASE_WIDTH + g.blending * BLEND_WIDTH;

    let shadow = 1.0 - smoothstep(midpoint - width, midpoint + width * 0.25, l);
    let highlight = smoothstep(midpoint - width * 0.25, midpoint + width, l);
    let midtone = (1.0 - shadow - highlight).max(0.0);
    // Global always applies at full strength.
    let weights = [shadow, midtone, highlight, 1.0];

    for (wheel, w) in g.wheels.iter().zip(weights) {
        p[1] += wheel[0] * w;
        p[2] += wheel[1] * w;
        p[0] += wheel[2] * w;
    }
}

/// OkLab hue angle (radians) of a fully saturated sRGB colour with HSV hue
/// `deg`, so a wheel's on-screen hue maps to the tint actually added.
fn srgb_hue_to_oklab_hue(deg: f32) -> f32 {
    let [r, g, b] = hsv_to_rgb(deg, 1.0, 1.0);
    let [_, a, bb] = rgb_to_oklab(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b));
    bb.atan2(a)
}

/// HSV → RGB, each component in `0..=1`. `h` is in degrees.
pub(crate) fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chroma(p: [f32; 3]) -> f32 {
        (p[1] * p[1] + p[2] * p[2]).sqrt()
    }

    #[test]
    fn identity_leaves_pixel_unchanged() {
        let g = ColorGrading::default();
        let params = GradingParams::new(&g);
        assert!(params.is_identity());
        let mut p = [0.4f32, 0.08, -0.05];
        let orig = p;
        apply_grading(&mut p, &params);
        assert_eq!(p, orig);
    }

    #[test]
    fn shadow_wheel_tints_shadows_more_than_highlights() {
        let mut g = ColorGrading::default();
        g.shadows = GradingWheel {
            hue: 220.0,
            sat: 80.0,
            lum: 0.0,
        };
        let params = GradingParams::new(&g);
        assert!(!params.is_identity());

        let mut dark = [0.2f32, 0.0, 0.0];
        let mut bright = [0.85f32, 0.0, 0.0];
        let dark_before = chroma(dark);
        let bright_before = chroma(bright);
        apply_grading(&mut dark, &params);
        apply_grading(&mut bright, &params);
        let dark_gain = chroma(dark) - dark_before;
        let bright_gain = chroma(bright) - bright_before;
        assert!(
            dark_gain > bright_gain,
            "shadows {dark_gain} vs highlights {bright_gain}"
        );
        assert!(bright_gain.abs() < 1e-4, "highlights should stay neutral");
    }

    #[test]
    fn global_wheel_tints_every_zone() {
        let mut g = ColorGrading::default();
        g.global = GradingWheel {
            hue: 30.0,
            sat: 60.0,
            lum: 0.0,
        };
        let params = GradingParams::new(&g);
        let mut dark = [0.15f32, 0.0, 0.0];
        let mut bright = [0.85f32, 0.0, 0.0];
        apply_grading(&mut dark, &params);
        apply_grading(&mut bright, &params);
        assert!(chroma(dark) > 0.0);
        assert!(chroma(bright) > 0.0);
    }

    #[test]
    fn luminance_shifts_lightness() {
        let mut g = ColorGrading::default();
        g.midtones = GradingWheel {
            hue: 0.0,
            sat: 0.0,
            lum: 80.0,
        };
        let params = GradingParams::new(&g);
        let mut mid = [0.55f32, 0.0, 0.0];
        apply_grading(&mut mid, &params);
        assert!(mid[0] > 0.55);
    }

    #[test]
    fn hsv_primaries() {
        let red = hsv_to_rgb(0.0, 1.0, 1.0);
        assert!(red[0] > 0.99 && red[1] < 0.01 && red[2] < 0.01);
        let green = hsv_to_rgb(120.0, 1.0, 1.0);
        assert!(green[1] > 0.99 && green[0] < 0.01);
        let blue = hsv_to_rgb(240.0, 1.0, 1.0);
        assert!(blue[2] > 0.99 && blue[0] < 0.01);
    }
}
