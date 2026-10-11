// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! Vignette: darkens (or lightens) the frame corners.
//!
//! Unlike the tone/colour controls, a vignette depends on the pixel's *position*,
//! not just its value, so it cannot be baked into the 3D LUT. It runs in the
//! spatial pass, last (after grain), on display-referred sRGB values.
//!
//! The falloff is a superellipse over the frame-centred coordinates:
//! `Roundness` blends the shape between the frame ellipse, a true circle
//! (positive) and a rectangle (negative); `Midpoint` moves the onset;
//! `Feather` softens the transition. `Amount < 0` darkens the corners,
//! `Amount > 0` lightens them toward white.
//!
//! The default shape and the amount-to-white behaviour follow Lightroom's
//! post-crop vignette; the implementation is TinyLuma's own.

use serde::{Deserialize, Serialize};

/// Vignette settings. `serde(default)` keeps older presets loading as a no-op.
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize, Debug)]
#[serde(default)]
pub(crate) struct VignetteSettings {
    /// Corner strength, `-100..=100` (`< 0` darkens, `> 0` lightens).
    pub(crate) amount: f32,
    /// Where the falloff reaches half strength, `0..=100` (default 50).
    pub(crate) midpoint: f32,
    /// Shape: `-100` rectangle, `0` frame ellipse, `+100` circle.
    pub(crate) roundness: f32,
    /// Transition softness, `0..=100` (default 50).
    pub(crate) feather: f32,
}

impl Default for VignetteSettings {
    fn default() -> Self {
        Self {
            amount: 0.0,
            midpoint: 50.0,
            roundness: 0.0,
            feather: 50.0,
        }
    }
}

impl VignetteSettings {
    /// Whether the vignette is a no-op.
    pub(crate) fn is_identity(&self) -> bool {
        self.amount.abs() < 1e-6
    }
}

/// Vignette resolved for the pixel loop: normalized amount, midpoint, width and
/// roundness, precomputed once per render.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VignetteParams {
    /// `-1..=1`.
    amount: f32,
    /// Distance of half strength.
    mid: f32,
    /// Half-width of the transition.
    width: f32,
    /// `-1..=1`.
    roundness: f32,
}

impl VignetteParams {
    pub(crate) fn new(v: &VignetteSettings) -> Self {
        Self {
            amount: (v.amount / 100.0).clamp(-1.0, 1.0),
            mid: 0.10 + (v.midpoint / 100.0).clamp(0.0, 1.0) * 0.75,
            width: 0.01 + (v.feather / 100.0).clamp(0.0, 1.0) * 0.40,
            roundness: (v.roundness / 100.0).clamp(-1.0, 1.0),
        }
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.amount.abs() < 1e-6
    }

    /// Falloff weight at pixel `(x, y)`: `0` at the centre, `1` in the corners.
    #[inline]
    pub(crate) fn mask(&self, x: usize, y: usize, width: usize, height: usize) -> f32 {
        let dx = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
        let dy = (y as f32 + 0.5) / height as f32 * 2.0 - 1.0;
        let d_ellipse = (dx * dx + dy * dy).sqrt();
        let r = self.roundness;
        // Roundness blends between the frame ellipse and a square/rectangle
        // (negative) or a true circle (positive).
        let d = if r >= 0.0 {
            let aspect = width as f32 / height as f32;
            let (cx, cy) = if aspect >= 1.0 {
                (dx, dy / aspect)
            } else {
                (dx * aspect, dy)
            };
            let d_circle = (cx * cx + cy * cy).sqrt();
            d_ellipse + (d_circle - d_ellipse) * r
        } else {
            let d_square = dx.abs().max(dy.abs());
            d_ellipse + (d_square - d_ellipse) * (-r)
        } * 0.5;

        smoothstep(self.mid - self.width, self.mid + self.width, d)
    }

    /// Applies the vignette to a display-referred sRGB pixel.
    #[inline]
    pub(crate) fn apply(&self, rgb: &mut [f32; 3], x: usize, y: usize, width: usize, height: usize) {
        if self.is_identity() {
            return;
        }
        let m = self.mask(x, y, width, height);
        if m <= 0.0 {
            return;
        }
        if self.amount < 0.0 {
            let f = 1.0 + self.amount * m;
            for c in rgb.iter_mut() {
                *c *= f;
            }
        } else {
            let a = self.amount * m;
            for c in rgb.iter_mut() {
                *c += (1.0 - *c).max(0.0) * a;
            }
        }
    }
}

#[inline]
fn smoothstep(x0: f32, x1: f32, x: f32) -> f32 {
    let t = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dark_amount(amount: f32) -> VignetteSettings {
        VignetteSettings {
            amount,
            ..VignetteSettings::default()
        }
    }

    #[test]
    fn identity_leaves_pixel_unchanged() {
        let params = VignetteParams::new(&VignetteSettings::default());
        assert!(params.is_identity());
        let mut rgb = [0.4f32, 0.5, 0.6];
        params.apply(&mut rgb, 10, 10, 100, 100);
        assert_eq!(rgb, [0.4, 0.5, 0.6]);
    }

    #[test]
    fn mask_grows_from_centre_to_corner() {
        let p = VignetteParams::new(&dark_amount(-50.0));
        let centre = p.mask(50, 50, 100, 100);
        let edge = p.mask(99, 50, 100, 100);
        let corner = p.mask(99, 99, 100, 100);
        assert!(centre < edge, "centre {centre} vs edge {edge}");
        assert!(edge < corner, "edge {edge} vs corner {corner}");
    }

    #[test]
    fn negative_amount_darkens_corners_only() {
        let p = VignetteParams::new(&dark_amount(-100.0));
        let mut centre = [0.5f32, 0.5, 0.5];
        let mut corner = [0.5f32, 0.5, 0.5];
        p.apply(&mut centre, 50, 50, 100, 100);
        p.apply(&mut corner, 99, 99, 100, 100);
        assert!((centre[0] - 0.5).abs() < 0.05, "centre changed: {centre:?}");
        assert!(corner[0] < 0.2, "corner not darkened: {corner:?}");
    }

    #[test]
    fn positive_amount_lightens_corners_toward_white() {
        let p = VignetteParams::new(&dark_amount(80.0));
        let mut corner = [0.2f32, 0.2, 0.2];
        let mut centre = [0.2f32, 0.2, 0.2];
        p.apply(&mut corner, 99, 99, 100, 100);
        p.apply(&mut centre, 50, 50, 100, 100);
        assert!(corner[0] > 0.5, "corner not lightened: {corner:?}");
        assert!((centre[0] - 0.2).abs() < 0.05, "centre changed: {centre:?}");
    }

    #[test]
    fn mask_is_resolution_independent() {
        // The same relative position gives (almost) the same mask at any scale;
        // the tiny difference is the half-pixel centre offset.
        let p = VignetteParams::new(&dark_amount(-50.0));
        let a = p.mask(120, 80, 480, 320);
        let b = p.mask(480, 320, 1920, 1280);
        assert!((a - b).abs() < 5e-3, "{a} vs {b}");
    }
}
