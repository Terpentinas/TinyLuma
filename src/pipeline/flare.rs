// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! Creative light flares.
//!
//! A lens flare is built from a bright-pass of the image: an eight-point
//! starburst, an anamorphic horizontal streak, mirrored "ghost" copies and a
//! soft radial glow. The map depends only on the input, so it is generated once
//! per render at a reduced resolution and cached in `SpatialBase`; moving the
//! amount slider only re-mixes the precomputed map.
//!
//! **Only scattered light is stored.** Every component is the positive
//! difference between the spread value and the source (`spread − source`), so a
//! flat bright area contributes nothing — otherwise a large highlight turned
//! into a hard white blob. The starburst/streak use a normalised directional
//! average, which cancels the source term exactly on flat regions.
//!
//! The look follows RapidRAW's `flare.wgsl` (AGPL-3.0): threshold → starburst
//! rays → radial glow → ghosts → horizontal streak, added back with a luminance
//! protection. Unlike the GPU original (per-pixel ray marching) the CPU port
//! builds the map with directional scans and small blurs, which is far cheaper
//! while keeping the same character.

use rayon::prelude::*;

use crate::color::oklab::{linear_to_srgb, srgb_to_linear};
use crate::pipeline::creative::{luma, perceptual, smoothstep};

/// Long side of the working grid the flare is computed on.
const WORK_MAX: usize = 220;
/// Bright-pass threshold (display luma) and soft knee.
const THRESHOLD: f32 = 0.70;
const KNEE: f32 = 0.18;
/// Directional scan length and per-step decay for the starburst / streak.
const STAR_STEPS: usize = 26;
const STAR_DECAY: f32 = 0.92;
/// Gain of the summed starburst rays.
const STAR_GAIN: f32 = 2.6;
const STREAK_STEPS: usize = 48;
const STREAK_DECAY: f32 = 0.955;
const STREAK_GAIN: f32 = 2.0;
/// Hard cap on the stored flare (before the amount is applied), so a very large
/// highlight can never produce a pure-white blob.
const FLARE_CAP: f32 = 1.0;
/// Maximum light added at amount 100 (cap × gain must stay below 1, so a flare
/// can brighten but never turn a dark pixel pure white).
const FLARE_GAIN: f32 = 0.65;

/// A precomputed flare map at working resolution, stored as linear RGB.
pub(crate) struct FlareBase {
    pub(crate) data: Vec<f32>,
    pub(crate) w: usize,
    pub(crate) h: usize,
}

impl FlareBase {
    /// Bilinear sample of the flare at full-resolution pixel `(x, y)`.
    pub(crate) fn sample(&self, x: usize, y: usize, w: usize, h: usize) -> [f32; 3] {
        let (sw, sh) = (self.w, self.h);
        let fx = (x as f32 + 0.5) * sw as f32 / w as f32 - 0.5;
        let fy = (y as f32 + 0.5) * sh as f32 / h as f32 - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let xi = |v: f32| (v as isize).clamp(0, sw as isize - 1) as usize;
        let yi = |v: f32| (v as isize).clamp(0, sh as isize - 1) as usize;
        let (xa, xb) = (xi(x0), xi(x0 + 1.0));
        let (ya, yb) = (yi(y0), yi(y0 + 1.0));
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let v00 = self.data[(ya * sw + xa) * 3 + c];
            let v10 = self.data[(ya * sw + xb) * 3 + c];
            let v01 = self.data[(yb * sw + xa) * 3 + c];
            let v11 = self.data[(yb * sw + xb) * 3 + c];
            let top = v00 * (1.0 - tx) + v10 * tx;
            let bot = v01 * (1.0 - tx) + v11 * tx;
            out[c] = top * (1.0 - ty) + bot * ty;
        }
        out
    }
}

/// Adds the flare onto a display-referred pixel. `amount` is normalized
/// (`0..=1`); `flare` is a linear sample from [`FlareBase`].
pub(crate) fn apply_flare(center_display: [f32; 3], flare: [f32; 3], amount: f32) -> [f32; 3] {
    if amount <= 0.0 {
        return center_display;
    }
    let c = [
        srgb_to_linear(center_display[0]),
        srgb_to_linear(center_display[1]),
        srgb_to_linear(center_display[2]),
    ];
    let f = [
        flare[0].clamp(0.0, FLARE_CAP),
        flare[1].clamp(0.0, FLARE_CAP),
        flare[2].clamp(0.0, FLARE_CAP),
    ];
    let pl = perceptual(luma(c));
    let protection = 1.0 - smoothstep(0.7, 1.8, pl);
    let k = amount * protection * FLARE_GAIN;
    [
        linear_to_srgb(c[0] + f[0] * k),
        linear_to_srgb(c[1] + f[1] * k),
        linear_to_srgb(c[2] + f[2] * k),
    ]
}

/// Builds the flare map from a display-referred RGB buffer.
pub(crate) fn generate(src: &[u8], w: usize, h: usize) -> FlareBase {
    let scale = (w.max(h) as f32 / WORK_MAX as f32).max(1.0);
    let fw = ((w as f32 / scale).round() as usize).max(1);
    let fh = ((h as f32 / scale).round() as usize).max(1);

    let bright = bright_pass(src, w, h, fw, fh);

    // Soft radial halo: the blurred bright-pass minus the source itself, so a
    // flat bright region yields nothing.
    let radius = (fw.min(fh) / 16).max(2);
    let glow_blur = blur_rgb(&bright, fw, fh, radius);
    let mut glow = vec![0.0f32; bright.len()];
    glow.par_iter_mut().enumerate().for_each(|(i, o)| {
        *o = (glow_blur[i] - bright[i]).max(0.0);
    });

    // Eight-point starburst: one directional scatter per axis, both directions.
    let axes = [(1isize, 0isize), (0, 1), (1, 1), (1, -1)];
    let mut star = vec![0.0f32; bright.len()];
    for (dx, dy) in axes {
        for sign in [1isize, -1] {
            let s = directional_scatter(
                &bright,
                fw,
                fh,
                dx * sign,
                dy * sign,
                STAR_STEPS,
                STAR_DECAY,
            );
            for (a, v) in star.iter_mut().zip(s.iter()) {
                *a += *v;
            }
        }
    }
    star.par_iter_mut().for_each(|v| *v *= STAR_GAIN);
    let star = blur_rgb(&star, fw, fh, 1);

    // Anamorphic horizontal streak.
    let mut streak = directional_scatter(&bright, fw, fh, 1, 0, STREAK_STEPS, STREAK_DECAY);
    streak.par_iter_mut().for_each(|v| *v *= STREAK_GAIN);
    let streak = blur_rgb(&streak, fw, fh, 1);

    // Mirrored ghosts (already source-subtracted).
    let ghost = ghosts(&bright, fw, fh);

    let mut data = vec![0.0f32; fw * fh * 3];
    data.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
        let star_v = [star[i * 3], star[i * 3 + 1], star[i * 3 + 2]];
        let streak_v = [streak[i * 3], streak[i * 3 + 1], streak[i * 3 + 2]];
        let glow_v = [glow[i * 3], glow[i * 3 + 1], glow[i * 3 + 2]];
        let ghost_v = [ghost[i * 3], ghost[i * 3 + 1], ghost[i * 3 + 2]];
        // Warm starburst, cooler streak (anamorphic flares usually lean blue),
        // neutral halo and ghosts.
        let combine = |s: f32, st: f32, g: f32, gh: f32, tint: f32| {
            (s + st * 0.55 + g * 0.45 + gh * 0.8) * tint
        };
        o[0] = combine(star_v[0], streak_v[0], glow_v[0], ghost_v[0], 1.05).min(FLARE_CAP);
        o[1] = combine(star_v[1], streak_v[1], glow_v[1], ghost_v[1], 1.0).min(FLARE_CAP);
        o[2] = combine(star_v[2], streak_v[2], glow_v[2], ghost_v[2], 0.95).min(FLARE_CAP);
    });

    FlareBase { data, w: fw, h: fh }
}

/// Box-downsamples and thresholds the source into a linear bright-pass map.
fn bright_pass(src: &[u8], w: usize, h: usize, fw: usize, fh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; fw * fh * 3];
    let sx = w as f32 / fw as f32;
    let sy = h as f32 / fh as f32;
    out.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
        let dx = i % fw;
        let dy = i / fw;
        let x0 = (dx as f32 * sx) as usize;
        let x1 = (((dx + 1) as f32 * sx) as usize).clamp(x0 + 1, w);
        let y0 = (dy as f32 * sy) as usize;
        let y1 = (((dy + 1) as f32 * sy) as usize).clamp(y0 + 1, h);
        let mut acc = [0.0f32; 3];
        let mut cnt = 0.0f32;
        for y in y0..y1 {
            for x in x0..x1 {
                let b = (y * w + x) * 3;
                acc[0] += srgb_to_linear(src[b] as f32 / 255.0);
                acc[1] += srgb_to_linear(src[b + 1] as f32 / 255.0);
                acc[2] += srgb_to_linear(src[b + 2] as f32 / 255.0);
                cnt += 1.0;
            }
        }
        let inv = 1.0 / cnt.max(1.0);
        acc[0] *= inv;
        acc[1] *= inv;
        acc[2] *= inv;
        let l = luma(acc);
        let k = smoothstep(THRESHOLD, THRESHOLD + KNEE, l);
        o[0] = acc[0] * k;
        o[1] = acc[1] * k;
        o[2] = acc[2] * k;
    });
    out
}

/// Normalised directional average minus the source: the light that a bright
/// feature spreads along `(dx, dy)`. A flat region cancels exactly to zero, so
/// only edges and point highlights contribute.
fn directional_scatter(
    src: &[f32],
    w: usize,
    h: usize,
    dx: isize,
    dy: isize,
    steps: usize,
    decay: f32,
) -> Vec<f32> {
    let n = src.len();
    let mut conv = src.to_vec();
    let mut carry = src.to_vec();
    let mut weight = 1.0f32;
    let mut total = 1.0f32;
    for _ in 0..steps {
        let mut shifted = vec![0.0f32; n];
        shifted.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
            let x = (i % w) as isize;
            let y = (i / w) as isize;
            let sx = x - dx;
            let sy = y - dy;
            if sx >= 0 && sx < w as isize && sy >= 0 && sy < h as isize {
                let j = sy as usize * w + sx as usize;
                o[0] = carry[j * 3];
                o[1] = carry[j * 3 + 1];
                o[2] = carry[j * 3 + 2];
            }
        });
        weight *= decay;
        total += weight;
        conv.par_iter_mut()
            .zip(shifted.par_iter())
            .for_each(|(c, s)| {
                *c += weight * s;
            });
        carry = shifted;
    }
    let inv = 1.0 / total.max(1.0);
    conv.par_iter_mut()
        .zip(src.par_iter())
        .for_each(|(c, s)| *c = (*c * inv - *s).max(0.0));
    conv
}

/// Separable Gaussian blur of an interleaved RGB plane (σ ≈ radius).
fn blur_rgb(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    if radius == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let kernel = gaussian_kernel(radius);
    let r = radius as isize;
    let mut tmp = vec![0.0f32; src.len()];
    tmp.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
        let x = (i % w) as isize;
        let y = (i / w) as isize;
        let row = y as usize * w;
        let mut acc = [0.0f32; 3];
        for (k, &kv) in kernel.iter().enumerate() {
            let sx = (x + k as isize - r).clamp(0, w as isize - 1) as usize;
            let j = row + sx;
            acc[0] += src[j * 3] * kv;
            acc[1] += src[j * 3 + 1] * kv;
            acc[2] += src[j * 3 + 2] * kv;
        }
        o.copy_from_slice(&acc);
    });
    let mut out = vec![0.0f32; src.len()];
    out.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
        let x = i % w;
        let y = (i / w) as isize;
        let mut acc = [0.0f32; 3];
        for (k, &kv) in kernel.iter().enumerate() {
            let sy = (y + k as isize - r).clamp(0, h as isize - 1) as usize;
            let j = sy * w + x;
            acc[0] += tmp[j * 3] * kv;
            acc[1] += tmp[j * 3 + 1] * kv;
            acc[2] += tmp[j * 3 + 2] * kv;
        }
        o.copy_from_slice(&acc);
    });
    out
}

fn gaussian_kernel(radius: usize) -> Vec<f32> {
    let radius = radius.max(1);
    let sigma = radius as f32;
    let mut k = vec![0.0f32; 2 * radius + 1];
    let mut sum = 0.0f32;
    for (i, v) in k.iter_mut().enumerate() {
        let d = i as f32 - radius as f32;
        *v = (-(d * d) / (2.0 * sigma * sigma)).exp();
        sum += *v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Mirrored, scaled-down ghosts of the bright pass, each with a vignette.
/// The local source is subtracted so a bright area does not just add itself
/// back as a veil.
fn ghosts(bright: &[f32], w: usize, h: usize) -> Vec<f32> {
    const SCALES: [f32; 4] = [0.75, 0.40, 0.20, 0.12];
    const WEIGHTS: [f32; 4] = [0.95, 1.15, 1.25, 1.05];
    let mut acc = vec![0.0f32; bright.len()];
    for (scale, &weight) in SCALES.iter().zip(WEIGHTS.iter()) {
        let s = *scale;
        acc.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
            let x = i % w;
            let y = i / w;
            let u = (x as f32 + 0.5) / w as f32;
            let v = (y as f32 + 0.5) / h as f32;
            // Mirror around the centre, then shrink toward it.
            let su = 0.5 + ((1.0 - u) - 0.5) * s;
            let sv = 0.5 + ((1.0 - v) - 0.5) * s;
            let sample = sample_bilinear(bright, w, h, su, sv);
            // Vignette: ghosts live away from the centre.
            let du = (su - 0.5) * 2.0;
            let dv = (sv - 0.5) * 2.0;
            let dist = (du * du + dv * dv).sqrt();
            let vig = 1.0 - smoothstep(0.15, 0.6, dist);
            let k = weight * vig * 0.12;
            for c in 0..3 {
                let src = bright[i * 3 + c];
                o[c] += (sample[c] - src).max(0.0) * k;
            }
        });
    }
    acc
}

/// Bilinear sample of a normalized interleaved RGB map.
fn sample_bilinear(data: &[f32], w: usize, h: usize, u: f32, v: f32) -> [f32; 3] {
    let fx = u.clamp(0.0, 1.0) * (w as f32 - 1.0);
    let fy = v.clamp(0.0, 1.0) * (h as f32 - 1.0);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let v00 = data[(y0 * w + x0) * 3 + c];
        let v10 = data[(y0 * w + x1) * 3 + c];
        let v01 = data[(y1 * w + x0) * 3 + c];
        let v11 = data[(y1 * w + x1) * 3 + c];
        let top = v00 * (1.0 - tx) + v10 * tx;
        let bot = v01 * (1.0 - tx) + v11 * tx;
        out[c] = top * (1.0 - ty) + bot * ty;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_image_has_no_flare() {
        let (w, h) = (32usize, 24usize);
        let src = vec![10u8; w * h * 3];
        let base = generate(&src, w, h);
        let total: f32 = base.data.iter().sum();
        assert!(total < 1e-3, "dark image produced flare: {total}");
    }

    #[test]
    fn flat_bright_image_has_no_flare() {
        // The regression this guards: a large bright area used to leak the
        // source and produce a hard white blob.
        let (w, h) = (64usize, 48usize);
        let src = vec![255u8; w * h * 3];
        let base = generate(&src, w, h);
        let max: f32 = base.data.iter().cloned().fold(0.0, f32::max);
        assert!(max < 0.05, "flat bright image flared: max {max}");
    }

    #[test]
    fn highlight_produces_flare() {
        let (w, h) = (48usize, 32usize);
        let mut src = vec![5u8; w * h * 3];
        // A bright spot near the centre.
        for y in 14..18 {
            for x in 22..26 {
                let b = (y * w + x) * 3;
                src[b] = 255;
                src[b + 1] = 255;
                src[b + 2] = 255;
            }
        }
        let base = generate(&src, w, h);
        let total: f32 = base.data.iter().sum();
        assert!(total > 0.01, "highlight produced no flare: {total}");
        let max: f32 = base.data.iter().cloned().fold(0.0, f32::max);
        assert!(max <= FLARE_CAP + 1e-6, "flare exceeded cap: {max}");
    }

    #[test]
    fn identity_at_zero_amount() {
        let c = [0.4f32, 0.5, 0.6];
        assert_eq!(apply_flare(c, [1.0, 0.5, 0.2], 0.0), c);
    }

    #[test]
    fn flare_adds_bounded_light() {
        let out = apply_flare([0.1, 0.1, 0.1], [FLARE_CAP, FLARE_CAP, FLARE_CAP], 1.0);
        assert!(out[0] > 0.1, "{out:?}");
        // A capped flare at full amount must stay below pure white on a dark base.
        assert!(out[0] < 1.0, "flare clipped to white: {out:?}");
    }
}
