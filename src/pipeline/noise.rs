// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! Noise reduction — pure model and per-pixel blend math.
//!
//! AI-generated images rarely carry sensor noise, but diffusion output often has
//! **chroma speckle** and low-frequency colour blotching, and low-step /
//! high-CFG renders can show luminance grain. This module owns the settings and
//! the pure blend math; the blurs (a wide chroma low-pass and an edge-aware
//! luminance guide) are precomputed in the spatial pass
//! (`app/processing.rs::compute_noise_base`) so slider drags only re-mix.
//!
//! Chroma is reduced in **Oklab** (`a`/`b` only, `L` untouched); luminance uses
//! a **guided-filter** guide and a log2 gain. Both are edge-aware: small
//! deviations (noise) are averaged toward the guide, large ones (real edges) are
//! kept, controlled by the sliders' Detail.
//!
//! Adapted from:
//! - lightcraft `crates/pipeline/src/local.rs` — guided-filter luma NR on
//!   log-luminance (MIT OR Apache-2.0)
//! - rawmakase `crates/rawmakase-engine/src/develop/color_noise.rs` —
//!   edge-aware chroma soft gain (MIT)
//! - RapidRAW `src-tauri/src/shaders/shader.wgsl` `apply_noise_reduction` —
//!   luma/chroma split (AGPL-3.0)

use serde::{Deserialize, Serialize};

/// Default fine-detail preservation (`50` = the neutral Lightroom-like value).
pub(crate) const NR_DEFAULT_DETAIL: f32 = 50.0;

/// Noise-reduction settings. `Copy`, so it lives in `FilterSettings`.
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize, Debug)]
pub(crate) struct NoiseSettings {
    /// Luminance noise reduction, `0..=100`. `0` disables it.
    #[serde(default)]
    pub(crate) luma: f32,
    /// Chroma (colour) noise reduction, `0..=100`. `0` disables it.
    #[serde(default)]
    pub(crate) chroma: f32,
    /// Fine-detail preservation, `0..=100`. Higher keeps more fine structure and
    /// larger colour/luma structures. `serde(default)` keeps older presets
    /// loading (and the value is neutral while luma/chroma are off).
    #[serde(default = "default_nr_detail")]
    pub(crate) detail: f32,
}

pub(crate) fn default_nr_detail() -> f32 {
    NR_DEFAULT_DETAIL
}

impl Default for NoiseSettings {
    fn default() -> Self {
        Self {
            luma: 0.0,
            chroma: 0.0,
            detail: NR_DEFAULT_DETAIL,
        }
    }
}

impl NoiseSettings {
    /// True when both reduction controls are off — the whole feature is a no-op
    /// regardless of Detail.
    pub(crate) fn is_identity(&self) -> bool {
        self.luma <= 0.0 && self.chroma <= 0.0
    }
}

/// Radius (px) of the wide chroma low-pass, at preview resolution. Scaled with
/// `detail_scale` by the spatial pass so preview and export agree.
pub(crate) const CHROMA_RADIUS: f32 = 4.0;
/// Radius (px) and epsilon of the luminance guide, at preview resolution.
pub(crate) const LUMA_RADIUS: f32 = 3.0;
pub(crate) const LUMA_EPS: f32 = 0.02;

/// Per-pixel blend parameters derived from the sliders (slider-independent
/// radius lives in the precomputed base).
#[derive(Clone, Copy)]
pub(crate) struct NrPlan {
    pub(crate) chroma: Option<ChromaPlan>,
    pub(crate) luma: Option<LumaPlan>,
}

#[derive(Clone, Copy)]
pub(crate) struct ChromaPlan {
    /// Maximum fraction of `(cur - blurred)` removed on flat noise, `0..=1`.
    pub(crate) mix: f32,
    /// Chroma difference (Oklab) above which a real colour edge is kept.
    pub(crate) threshold: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct LumaPlan {
    pub(crate) mix: f32,
    /// Log2-luminance difference above which a real edge is kept.
    pub(crate) threshold: f32,
}

/// Builds the blend plan from the settings. `is_identity` yields `None`s.
pub(crate) fn plan(settings: &NoiseSettings) -> NrPlan {
    let detail = (settings.detail / 100.0).clamp(0.0, 1.0);
    let chroma = (settings.chroma / 100.0).clamp(0.0, 1.0);
    let luma = (settings.luma / 100.0).clamp(0.0, 1.0);
    // A lower threshold with more Detail, so stronger colour/luma edges and
    // finer structures survive (higher Detail = keep more).
    let threshold = 0.12 - 0.10 * detail;
    NrPlan {
        chroma: (chroma > 0.0).then(|| ChromaPlan {
            mix: chroma.sqrt(),
            threshold,
        }),
        luma: (luma > 0.0).then(|| LumaPlan {
            mix: luma.sqrt(),
            threshold,
        }),
    }
}

/// Edge-aware soft gate: `0` for a flat sample, approaching `1` for a real edge.
#[inline]
fn soft_gate(x: f32) -> f32 {
    x * x / (1.0 + x * x)
}

/// Blends the current Oklab chroma `[a, b]` toward the blurred chroma. Small
/// differences (noise) move fully toward the blur; large ones (edges) stay.
#[inline]
pub(crate) fn blend_chroma(cur: [f32; 2], blurred: [f32; 2], p: &ChromaPlan) -> [f32; 2] {
    let da = cur[0] - blurred[0];
    let db = cur[1] - blurred[1];
    let x = (da * da + db * db).sqrt() / p.threshold.max(1e-6);
    let keep = (1.0 - p.mix) + p.mix * soft_gate(x);
    [blurred[0] + da * keep, blurred[1] + db * keep]
}

/// Multiplier applied to the pixel's RGB for luminance NR. `cur_log`/`guide_log`
/// are log2 luminances of the pixel and the edge-aware guide; flat areas are
/// pulled toward the guide, edges (where the guide already followed) are kept.
#[inline]
pub(crate) fn luma_gain(cur_log: f32, guide_log: f32, p: &LumaPlan) -> f32 {
    let detail = cur_log - guide_log;
    let x = detail.abs() / p.threshold.max(1e-6);
    let keep = (1.0 - p.mix) + p.mix * soft_gate(x);
    // new = guide + detail*keep  →  delta = detail*(keep - 1)
    (detail * (keep - 1.0)).exp2()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(luma: f32, chroma: f32, detail: f32) -> NoiseSettings {
        NoiseSettings {
            luma,
            chroma,
            detail,
        }
    }

    #[test]
    fn identity_when_both_off() {
        let s = settings(0.0, 0.0, 100.0);
        assert!(s.is_identity());
        let p = plan(&s);
        assert!(p.chroma.is_none() && p.luma.is_none());
    }

    #[test]
    fn plan_is_active_when_either_is_on() {
        assert!(plan(&settings(10.0, 0.0, 50.0)).luma.is_some());
        assert!(plan(&settings(0.0, 10.0, 50.0)).chroma.is_some());
    }

    #[test]
    fn chroma_noise_is_averaged_toward_the_blur() {
        let p = plan(&settings(0.0, 100.0, 50.0)).chroma.unwrap();
        // A tiny chroma deviation (noise) essentially disappears.
        let out = blend_chroma([0.10, 0.02], [0.10, 0.0], &p);
        assert!(out[1].abs() < 0.25 * 0.02, "noise chroma left: {}", out[1]);
    }

    #[test]
    fn chroma_edge_is_preserved() {
        let p = plan(&settings(0.0, 100.0, 50.0)).chroma.unwrap();
        // A large chroma difference (a real colour edge) is kept.
        let out = blend_chroma([0.30, 0.0], [0.0, 0.0], &p);
        assert!(out[0] > 0.9 * 0.30, "edge chroma lost: {}", out[0]);
    }

    #[test]
    fn luma_gain_is_one_when_off_and_smooths_noise() {
        let off = plan(&settings(0.0, 0.0, 50.0)).luma;
        assert!(off.is_none());

        let p = plan(&settings(100.0, 0.0, 50.0)).luma.unwrap();
        // A small luma deviation is damped toward the guide...
        let g = luma_gain(0.02, 0.0, &p);
        assert!(g < 1.0, "noise luma must be reduced, gain {g}");
        // ...while a strong edge is kept (gain ≈ 1).
        let edge = luma_gain(0.6, 0.0, &p);
        assert!(edge > 0.9, "edge luma must be kept, gain {edge}");
    }

    #[test]
    fn higher_detail_preserves_more_noise() {
        let low = plan(&settings(0.0, 100.0, 0.0)).chroma.unwrap();
        let high = plan(&settings(0.0, 100.0, 100.0)).chroma.unwrap();
        let d = 0.05_f32;
        let out_low = blend_chroma([0.0, d], [0.0, 0.0], &low);
        let out_high = blend_chroma([0.0, d], [0.0, 0.0], &high);
        // Detail lowers the threshold, so more of the structure is treated as
        // detail and kept (higher Detail = preserve more).
        assert!(out_high[1] > out_low[1]);
    }
}
