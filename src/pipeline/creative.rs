// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! Creative glow (bloom) and halation.
//!
//! Both take the image plus a precomputed low-pass reference and *add* light
//! back around bright areas. They are not per-pixel (the reference is a blur),
//! so they run in the spatial pass on display-referred sRGB values and convert
//! to linear light internally.
//!
//! The model follows RapidRAW's `apply_glow_bloom` / `apply_halation`
//! (`src-tauri/src/shaders/shader.wgsl`, AGPL-3.0): a perceptual-luma cutoff
//! picks the bright pixels, the blurred source provides the colour, and the
//! result is added with a luminance protection so already-bright cores do not
//! blow out. Glow is warm and neutral; halation is the red/orange film bleed.
//!
//! Light **flares** live in [`super::flare`].
//!
//! - Glow: light bloom around highlights, `0..=100`.
//! - Halation: red glow bleeding from the brightest highlights, `0..=100`.

use crate::color::oklab::{linear_to_srgb, srgb_to_linear};

/// Weight of the glow added at amount 100 (RapidRAW uses 3.8).
const GLOW_GAIN: f32 = 3.8;
/// Weight of the halation added at amount 100.
const HALATION_GAIN: f32 = 2.5;
/// Falloff range that maps the perceptual excess onto `0..=1`.
const GLOW_FALLOFF: f32 = 5.5;

/// Rec.709 luma.
#[inline]
pub(crate) fn luma(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Hermite smoothstep, clamped at both ends.
#[inline]
pub(crate) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Perceptual (gamma-2.2) luma, piecewise-linear above 1 so HDR values keep
/// ordering without exploding.
#[inline]
pub(crate) fn perceptual(l: f32) -> f32 {
    if l <= 1.0 {
        l.max(0.0).powf(1.0 / 2.2)
    } else {
        1.0 + (l - 1.0).powf(1.0 / 2.2)
    }
}

#[inline]
fn to_linear(display: [f32; 3]) -> [f32; 3] {
    [
        srgb_to_linear(display[0]),
        srgb_to_linear(display[1]),
        srgb_to_linear(display[2]),
    ]
}

#[inline]
fn to_display(linear: [f32; 3]) -> [f32; 3] {
    [
        linear_to_srgb(linear[0]),
        linear_to_srgb(linear[1]),
        linear_to_srgb(linear[2]),
    ]
}

/// Adds a warm bloom from `blurred_display` onto `center_display`.
///
/// `amount` is normalized (`0..=1`). Both inputs are display-referred sRGB.
/// Returns a display-referred colour (always finite; the caller clamps).
pub(crate) fn apply_glow(
    center_display: [f32; 3],
    blurred_display: [f32; 3],
    amount: f32,
) -> [f32; 3] {
    if amount <= 0.0 {
        return center_display;
    }
    let c = to_linear(center_display);
    let b = to_linear(blurred_display);
    let linear_luma = luma(b);
    if linear_luma <= 1e-5 {
        return center_display;
    }

    let pl = perceptual(linear_luma);
    // At amount 100 the cutoff drops to 0.08, so even midtones bloom.
    let cutoff = 0.75 + (0.08 - 0.75) * amount.clamp(0.0, 1.0);
    let cutoff_fade = smoothstep(cutoff, cutoff + 0.15, pl);
    if cutoff_fade <= 0.0 {
        return center_display;
    }

    let excess = (pl - cutoff).max(0.0);
    let normalized = excess / GLOW_FALLOFF;
    let bloom_intensity = smoothstep(0.0, 1.0, normalized).powf(0.45);

    // The blurred colour carries the hue; a slight warm tint matches film bloom.
    let bloom_color = if linear_luma > 0.01 {
        let warm = [1.03f32, 1.0, 0.97];
        [
            b[0] / linear_luma * warm[0],
            b[1] / linear_luma * warm[1],
            b[2] / linear_luma * warm[2],
        ]
    } else {
        [1.0, 0.99, 0.98]
    };

    let luma_factor = linear_luma.powf(0.6);
    let black_gate = smoothstep(0.0, 0.5, linear_luma).sqrt();

    let current_luma = luma(c);
    let protection = 1.0 - smoothstep(1.0, 2.2, current_luma);

    let k =
        bloom_intensity * luma_factor * cutoff_fade * black_gate * amount * GLOW_GAIN * protection;

    to_display([
        c[0] + bloom_color[0] * k,
        c[1] + bloom_color[1] * k,
        c[2] + bloom_color[2] * k,
    ])
}

/// Adds a red/orange halation bleed from `blurred_display` onto `center_display`.
///
/// `amount` is normalized (`0..=1`); both colours are display-referred sRGB.
pub(crate) fn apply_halation(
    center_display: [f32; 3],
    blurred_display: [f32; 3],
    amount: f32,
) -> [f32; 3] {
    if amount <= 0.0 {
        return center_display;
    }
    let c = to_linear(center_display);
    let b = to_linear(blurred_display);
    let linear_luma = luma(b);
    if linear_luma <= 1e-5 {
        return center_display;
    }

    let pl = perceptual(linear_luma);
    let cutoff = 0.85 + (0.1 - 0.85) * amount.clamp(0.0, 1.0);
    if pl <= cutoff {
        return center_display;
    }

    let excess = pl - cutoff;
    let range = (1.5 - cutoff).max(0.1);
    let mask = smoothstep(0.0, range * 0.6, excess);

    // Deep red core fading to an orange fringe.
    let core = [1.0f32, 0.15, 0.03];
    let fringe = [1.0f32, 0.32, 0.10];
    let blend = smoothstep(0.0, 0.7, mask);
    let tint = [
        fringe[0] + (core[0] - fringe[0]) * blend,
        fringe[1] + (core[1] - fringe[1]) * blend,
        fringe[2] + (core[2] - fringe[2]) * blend,
    ];

    let glow = [
        tint[0] * mask * linear_luma,
        tint[1] * mask * linear_luma,
        tint[2] * mask * linear_luma,
    ];

    let color_luma = luma(c);
    let desat = mask * 0.12;
    let affected = [
        c[0] + (color_luma - c[0]) * desat,
        c[1] + (color_luma - c[1]) * desat,
        c[2] + (color_luma - c[2]) * desat,
    ];

    // A touch of contrast reduction inside the bleed (film halation lifts blacks).
    let contrast = 1.0 - mask * 0.06;
    let base = [
        0.5 + (affected[0] - 0.5) * contrast,
        0.5 + (affected[1] - 0.5) * contrast,
        0.5 + (affected[2] - 0.5) * contrast,
    ];

    let k = amount * HALATION_GAIN;
    to_display([
        base[0] + glow[0] * k,
        base[1] + glow[1] * k,
        base[2] + glow[2] * k,
    ])
}

/// Broad local contrast ("Structure").
///
/// Lifts or lowers the difference between the pixel's luma and a wide low-pass
/// reference, producing the HDR-like midtone punch that sits below Clarity's
/// scale. `amount` is normalized (`-1..=1`); `base_luma` is the wide reference.
#[inline]
pub(crate) fn apply_structure(luma_now: f32, base_luma: f32, amount: f32) -> f32 {
    if amount.abs() < 1e-6 {
        return luma_now;
    }
    let diff = luma_now - base_luma;
    // Soft-clip the difference so hard edges do not turn into halos.
    let safe = diff / (1.0 + diff.abs() * 6.0);
    (luma_now + safe * amount * 2.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glow_identity_at_zero() {
        let c = [0.8f32, 0.7, 0.6];
        assert_eq!(apply_glow(c, [1.0, 1.0, 1.0], 0.0), c);
    }

    #[test]
    fn glow_brightens_near_highlights() {
        // A dark pixel next to a blown highlight gains light.
        let out = apply_glow([0.02, 0.02, 0.02], [1.0, 1.0, 1.0], 1.0);
        assert!(out[0] > 0.02, "glow did not brighten: {out:?}");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite glow: {out:?}"
        );
    }

    #[test]
    fn glow_leaves_dark_scene_alone() {
        // Nothing bright in the blurred reference → no bloom.
        let c = [0.1f32, 0.1, 0.1];
        let out = apply_glow(c, [0.05, 0.05, 0.05], 1.0);
        for i in 0..3 {
            assert!((out[i] - c[i]).abs() < 1e-6, "{out:?}");
        }
    }

    #[test]
    fn halation_identity_at_zero() {
        let c = [0.8f32, 0.7, 0.6];
        assert_eq!(apply_halation(c, [1.0, 1.0, 1.0], 0.0), c);
    }

    #[test]
    fn halation_adds_red_not_blue() {
        let out = apply_halation([0.2, 0.2, 0.2], [1.0, 1.0, 1.0], 1.0);
        assert!(out[0] > out[2], "halation should be red-dominant: {out:?}");
    }

    #[test]
    fn structure_increases_local_contrast() {
        // A pixel brighter than its surroundings is pushed further up.
        let bright = apply_structure(0.6, 0.5, 0.8);
        let dark = apply_structure(0.4, 0.5, 0.8);
        assert!(bright > 0.6, "{bright}");
        assert!(dark < 0.4, "{dark}");
    }

    #[test]
    fn structure_identity_at_zero() {
        assert_eq!(apply_structure(0.42, 0.1, 0.0), 0.42);
    }
}
