// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

use eframe::egui;
use image::RgbImage;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use super::proxy_job::{ProxyReply, ProxyRequest};
use super::TinyLumaApp;
use crate::color::oklab::{
    linear_to_srgb, oklab_to_linear_rgb, rgb_to_oklab, srgb_to_linear, srgb_u8_linear_table,
};
use crate::lut::Lut3D;
use crate::pipeline::color::{
    ColorSettings, WhiteBalance, apply_chroma, apply_hue_rotation, gamut_map_linear,
};
use crate::pipeline::creative;
use crate::pipeline::curve::CurveTables;
use crate::pipeline::flare::{self, FlareBase};
use crate::pipeline::grading::{GradingParams, apply_grading};
use crate::pipeline::hsl::apply_hsl;
use crate::pipeline::light::{
    LightSettings, apply_light, contrast_from_slider, exposure_from_slider, unit_from_slider,
};
use crate::pipeline::noise;
use crate::pipeline::vignette::VignetteParams;
use crate::retouch::RetouchLayer;
use crate::settings::FilterSettings;

/// Reference resolution for the detail group (matches the preview cap, 1200px).
///
/// The Sharpen/Texture/Clarity radii are normalized to it, so the preview
/// (≤1200px on the long side) and the full-size export produce the same
/// relative effect.
const DETAIL_WORK_MAX: usize = 1200;
/// Downsample factor for Clarity at the reference resolution.
const CLARITY_SCALE: usize = 8;

/// Radius (px, at reduced resolution) of the shared detail mask's high-pass and
/// its energy blur.
const MASK_RADIUS: f32 = 2.0;
/// Energy knee (luma²) of the detail mask: below it an area counts as flat.
const MASK_KNEE2: f32 = 0.002;

/// Long side of the reduced-resolution plane shared by Glow and Structure.
const CREATIVE_WORK_MAX: usize = 256;
/// Radius of the wide Gaussian blur, as a fraction of the base's short side
/// (RapidRAW blurs with radius 40 at a 1080px reference).
const GLOW_RADIUS_FRAC: f32 = 0.037;
/// Long side and radius fraction of the tighter Halation blur.
const HALATION_WORK_MAX: usize = 512;
const HALATION_RADIUS_FRAC: f32 = 0.0074;

/// Long side of the reduced-resolution interactive drag tier. While a
/// spatial slider is dragged the pipeline runs at this size (~4x fewer pixels at
/// the 1200px preview cap) and the smaller texture is stretched to the same
/// on-screen size; the authoritative full-resolution render happens on release.
const DRAG_WORK_MAX: usize = 640;

/// Preview LUT resolution (drag *and* release). A 33³ grid is ~1/255 mean error,
/// versus 64³ which only wins in the clipping worst case while costing ~18 ms on
/// release. Export stays at 64³ ([`TinyLumaApp::bake_temp_lut`]) so files do not
/// lose quality. Shared by the UI preview ([`TinyLumaApp::process_preview`]) and
/// the background proxy render ([`TinyLumaApp::render_preview_at`]) so both tiers
/// bake the same grid.
const PREVIEW_LUT_SIZE: usize = 33;

/// Resolution ladder for the interactive working proxy (long side, px).
/// The proxy is sized to roughly the physical pixels the image covers on screen
/// (`viewport · zoom · pixels_per_point`), snapped *up* to the next rung so a
/// zoom change rebuilds the proxy at most once per rung (~1.5x apart) instead of
/// on every wheel tick. Rung 0 doubles as the proxy floor.
const PROXY_LADDER: [usize; 6] = [640, 960, 1280, 1920, 2560, 4096];
/// Lower bound of the proxy long side. Small sources/viewports are never
/// upscaled past this; it matches the drag tier so the two tiers compose cleanly.
const PROXY_WORK_MIN: usize = DRAG_WORK_MAX;
/// Highest resolution the interactive proxy *source* is decoded at. A
/// larger image keeps a source at this cap — the proxy never exceeds it — while
/// the export path still uses the full-size source.
pub(crate) const SOURCE_WORK_MAX: usize = 4096;

/// How long the viewport proxy waits, once the desired resolution stops
/// changing (a zoom or window-size gesture), before rebuilding — so a fast wheel
/// zoom rebuilds the proxy once instead of every tick.
pub(crate) const PROXY_SETTLE: std::time::Duration = std::time::Duration::from_millis(180);

/// RGB texture base at working resolution (interleaved storage).
pub(crate) struct TextureBase {
    pub(crate) data: Vec<f32>,
    pub(crate) sw: usize,
    pub(crate) sh: usize,
}

/// Reduced-resolution planes for noise reduction: `[Oklab a, Oklab b, log2
/// guided luminance]` interleaved, sampled bilinearly by the spatial pass.
/// A wide chroma low-pass plus an edge-aware luminance guide.
pub(crate) struct NoiseBase {
    pub(crate) data: Vec<f32>,
    pub(crate) sw: usize,
    pub(crate) sh: usize,
}

/// Reduced-resolution per-pixel detail mask (`0..=1`): ~0 on flat/noisy areas,
/// ~1 on textured areas/edges. Shared by Texture, Clarity and Sharpen.
pub(crate) struct MaskBase {
    pub(crate) data: Vec<f32>,
    pub(crate) sw: usize,
    pub(crate) sh: usize,
}

/// Precomputations for the spatial pass, depending only on the input (color/luma) and
/// resolution, but NOT on the Texture/Clarity/Sharpen/Grain slider values.
/// Cached between frames so `gaussian_blur` and the Texture base are not computed
/// again on every slider move.
#[derive(Default)]
pub(crate) struct SpatialBase {
    pub(crate) sharpen_blur: Option<Vec<f32>>,
    pub(crate) texture_base: Option<TextureBase>,
    pub(crate) noise_base: Option<NoiseBase>,
    /// Shared detail mask for Texture/Clarity/Sharpen (present only when
    /// `detail_masking > 0`).
    pub(crate) mask_base: Option<MaskBase>,
    /// Wide RGB low-pass shared by Glow and Structure (present only when either
    /// is engaged). Stored as a reduced-resolution [`TextureBase`].
    pub(crate) wide_blur: Option<TextureBase>,
    /// Tighter RGB low-pass for Halation (present only when `halation > 0`).
    pub(crate) halation_blur: Option<TextureBase>,
    /// Precomputed light-flare map (present only when `flare > 0`).
    pub(crate) flare_base: Option<FlareBase>,
}

/// Scalar per-pixel parameters of the spatial pass, derived once per
/// pass from the settings (so the staged detail/effects halves do not re-derive
/// them per pixel).
struct SpatialParams {
    n_texture: f32,
    n_clarity: f32,
    n_sharpen: f32,
    n_masking: f32,
    n_structure: f32,
    n_glow: f32,
    n_halation: f32,
    n_flare: f32,
    nr_plan: noise::NrPlan,
    noise_active: bool,
    vignette: VignetteParams,
    vignette_active: bool,
    grain_amount: f32,
    grain_roughness: f32,
}

impl SpatialParams {
    fn new(s: &FilterSettings) -> Self {
        let vignette = VignetteParams::new(&s.vignette);
        Self {
            n_texture: s.texture / 100.0, // -1.0 .. 1.0
            n_clarity: s.clarity / 100.0, // -1.0 .. 1.0
            n_sharpen: s.sharpen / 100.0, // 0.0 .. 1.5
            n_masking: (s.detail_masking / 100.0).clamp(0.0, 1.0),
            n_structure: s.structure / 100.0, // -1.0 .. 1.0
            n_glow: s.glow / 100.0,           // 0.0 .. 1.0
            n_halation: s.halation / 100.0,   // 0.0 .. 1.0
            n_flare: s.flare / 100.0,         // 0.0 .. 1.0
            nr_plan: noise::plan(&s.noise),
            noise_active: !s.noise.is_identity(),
            vignette_active: !vignette.is_identity(),
            vignette,
            grain_amount: s.grain,
            grain_roughness: s.grain_roughness,
        }
    }
}

/// Persistent state for the reduced-resolution interactive drag tier.
///
/// The downscaled input and its precomputations (luma, clarity, spatial bases,
/// grain) are rebuilt only when the input revision, the drag resolution or the
/// base-affecting settings change (`tier_rev`). The f32 detail head is rebuilt
/// only when its own signature changes (`detail_sig`), so an effects-only drag
/// reuses it and skips the expensive detail work.
#[derive(Default)]
pub(crate) struct DragTier {
    pub(crate) input: Vec<u8>,
    pub(crate) clarity: Vec<f32>,
    pub(crate) clarity_sw: usize,
    pub(crate) clarity_sh: usize,
    pub(crate) spatial_base: SpatialBase,
    pub(crate) grain: Vec<f32>,
    pub(crate) grain_dims: (usize, usize),
    pub(crate) grain_key: (u32, bool),
    pub(crate) grain_valid: bool,
    /// f32 output of the detail head.
    pub(crate) detail: Vec<f32>,
    /// Signature the detail head was built for.
    pub(crate) detail_sig: u64,
    pub(crate) detail_valid: bool,
    /// `tier_rev` the input/base/clarity caches were built for.
    pub(crate) tier_rev: u64,
    pub(crate) tier_valid: bool,
    /// Final u8 drag output uploaded to the display texture.
    pub(crate) output: Vec<u8>,
}

impl TinyLumaApp {
    /// Pass 1: color. All edits are per-pixel and baked into the combined LUT.
    ///
    /// Order: sRGB→linear → white balance (Bradford) → Oklab → Light →
    /// HSL mixer → Hue → Saturation/Vibrance → Color Grading → soft gamut →
    /// linear→sRGB → external LUT.
    pub(crate) fn run_color_pass(
        settings: &FilterSettings,
        active_lut: Option<&Lut3D>,
        buffer: &mut [u8],
    ) {
        // Precompute the parameters and the table once for the whole buffer.
        let lin_lut = srgb_u8_linear_table();

        let wb = WhiteBalance::new(settings.temp, settings.tint);

        let light = LightSettings {
            exposure: exposure_from_slider(settings.exposure),
            contrast: contrast_from_slider(settings.contrast),
            highlights: unit_from_slider(settings.highlights),
            shadows: unit_from_slider(settings.shadows),
            whites: unit_from_slider(settings.whites),
            blacks: unit_from_slider(settings.blacks),
            brightness: unit_from_slider(settings.brightness),
        };

        let color = ColorSettings {
            saturation: unit_from_slider(settings.saturation),
            vibrance: unit_from_slider(settings.vibrance),
        };

        let wb_identity = wb.is_identity();
        let light_identity = light.is_identity();
        let hsl_identity = settings.hsl.is_identity();
        let hsl = settings.hsl;
        let chroma_identity = color.saturation.abs() < 1e-6 && color.vibrance.abs() < 1e-6;
        // Global hue rotation (radians). Applied after the mixer, before chroma,
        // matching RapidRAW's pipeline order.
        let hue = settings.hue.to_radians();
        let hue_identity = hue.abs() < 1e-6;

        // Color grading is per-pixel too (constant Oklab offsets), so it bakes
        // into the same LUT. Precompute the wheel offsets once.
        let grading = GradingParams::new(&settings.grading);
        let grading_identity = grading.is_identity();

        let lut_intensity = settings.lut_intensity / 100.0;

        // Tone curves are per-pixel too, so they bake into the same LUT. Composed
        // per channel (master, then R/G/B) into 256-sample tables once per call.
        let curves = settings.curves.bake();
        let curves_identity = settings.curves.is_identity();

        buffer.par_chunks_mut(3).for_each(|pixel| {
            let mut lin = [
                lin_lut[pixel[0] as usize],
                lin_lut[pixel[1] as usize],
                lin_lut[pixel[2] as usize],
            ];

            if !wb_identity {
                wb.apply_linear(&mut lin);
            }

            let mut p = rgb_to_oklab(lin[0], lin[1], lin[2]);

            if !light_identity {
                apply_light(&mut p, &light);
            }
            // Colour mixer sits between Light and chroma so a tone change also
            // changes how saturated the shifted hue reads.
            if !hsl_identity {
                apply_hsl(&mut p, &hsl);
            }
            // Global hue rotation sits between the mixer and chroma, so the
            // mixer works on the original hues and saturation reacts to the
            // rotated result.
            if !hue_identity {
                apply_hue_rotation(&mut p, hue);
            }
            if !chroma_identity {
                apply_chroma(&mut p, &color);
            }
            // Color grading tints by tonal zone after the mixer/chroma so it
            // reacts to the final lightness.
            if !grading_identity {
                apply_grading(&mut p, &grading);
            }

            let out = gamut_map_linear(p[0], p[1], p[2]);
            let mut r = linear_to_srgb(out[0]);
            let mut g = linear_to_srgb(out[1]);
            let mut b = linear_to_srgb(out[2]);

            // Tone curve on display-referred values, before the creative LUT.
            if !curves_identity {
                r = CurveTables::read(&curves.r, r);
                g = CurveTables::read(&curves.g, g);
                b = CurveTables::read(&curves.b, b);
            }

            if let Some(lut) = active_lut {
                if lut_intensity > 0.0 {
                    let mapped = lut.apply(r, g, b);
                    r = r * (1.0 - lut_intensity) + mapped[0] * lut_intensity;
                    g = g * (1.0 - lut_intensity) + mapped[1] * lut_intensity;
                    b = b * (1.0 - lut_intensity) + mapped[2] * lut_intensity;
                }
            }

            pixel[0] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            pixel[1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            pixel[2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        });
    }

    // ==========================================
    // BAKE: 64³ COMBINED LUT (all sliders + external .cube)
    // ==========================================
    pub(crate) fn bake_combined_lut(&self, size: usize) -> Option<Lut3D> {
        Some(Self::bake_lut(&self.settings, self.active_lut.as_deref(), size))
    }

    /// Bakes the combined `size³` LUT (all sliders + external `.cube`).
    ///
    /// Single entry point for the preview (`bake_combined_lut`), export
    /// (`bake_temp_lut`) and benchmarks: all the math goes through
    /// `run_color_pass` over an identity grid.
    pub(crate) fn bake_lut(
        settings: &FilterSettings,
        active_lut: Option<&Lut3D>,
        size: usize,
    ) -> Lut3D {
        let total = size * size * size;
        let mut buffer = vec![0u8; total * 3];

        // Fill the identity grid: the pixel color = its coordinate in the LUT.
        for iz in 0..size {
            for iy in 0..size {
                for ix in 0..size {
                    let idx = (ix + iy * size + iz * size * size) * 3;
                    buffer[idx] = (ix as f32 / (size - 1) as f32 * 255.0) as u8;
                    buffer[idx + 1] = (iy as f32 / (size - 1) as f32 * 255.0) as u8;
                    buffer[idx + 2] = (iz as f32 / (size - 1) as f32 * 255.0) as u8;
                }
            }
        }

        // All the slider math + the external LUT.
        Self::run_color_pass(settings, active_lut, &mut buffer);

        // Convert u8 → f32 Lut3D.
        let mut data = Vec::with_capacity(total);
        for chunk in buffer.chunks_exact(3) {
            data.push([
                chunk[0] as f32 / 255.0,
                chunk[1] as f32 / 255.0,
                chunk[2] as f32 / 255.0,
            ]);
        }

        Lut3D { size, data }
    }

    // ==========================================
    // PASS 2: SPATIAL EFFECTS
    // (Noise, Texture, Clarity, Sharpen, Grain, Vignette)
    // ==========================================
    /// Precomputations for the spatial pass (the bilateral Texture base and the luma
    /// blur for Sharpen). Depend ONLY on the input and resolution, but NOT on the
    /// slider values — so they can be cached and reused.
    pub(crate) fn compute_spatial_base(
        settings: &FilterSettings,
        color_buffer: &[u8],
        luma_map: &[f32],
        width: usize,
        height: usize,
    ) -> SpatialBase {
        let detail_scale = Self::detail_scale(width, height);
        let n_texture = settings.texture / 100.0;
        let n_sharpen = settings.sharpen / 100.0;

        // Sharpen: luma blur (separable Gaussian) with a fractional radius
        // proportional to the resolution, so the effect is scale-invariant.
        let sharpen_blur = if n_sharpen > 0.0 {
            Some(Self::gaussian_blur_map(
                luma_map,
                width,
                height,
                detail_scale,
            ))
        } else {
            None
        };

        // Texture: base (bilateral blur) at reduced resolution.
        let texture_base = if n_texture != 0.0 {
            Some(Self::compute_texture_base(
                color_buffer,
                luma_map,
                width,
                height,
                detail_scale,
            ))
        } else {
            None
        };

        // Noise reduction: wide chroma low-pass + edge-aware luminance guide,
        // both at reduced resolution. Radius is slider-independent (it only
        // scales with the resolution) so slider drags re-mix without a re-blur.
        let noise_base = (!settings.noise.is_identity())
            .then(|| Self::compute_noise_base(color_buffer, luma_map, width, height, detail_scale));

        // Shared detail mask for Texture/Clarity/Sharpen. Computed once from the
        // (reduced-resolution) luma high-pass, only while masking is on.
        let mask_base = (settings.detail_masking > 0.0)
            .then(|| Self::compute_mask_base(luma_map, width, height, detail_scale));

        // Wide low-pass shared by Glow and Structure. Its radius scales with the
        // base's short side, so preview and export match.
        let wide_blur = (settings.glow > 0.0 || settings.structure != 0.0).then(|| {
            Self::compute_blur_base(
                color_buffer,
                width,
                height,
                CREATIVE_WORK_MAX,
                GLOW_RADIUS_FRAC,
            )
        });

        // Tighter low-pass for Halation (red bleed stays close to the highlight).
        let halation_blur = (settings.halation > 0.0).then(|| {
            Self::compute_blur_base(
                color_buffer,
                width,
                height,
                HALATION_WORK_MAX,
                HALATION_RADIUS_FRAC,
            )
        });

        // Light-flare map: depends only on the input, so it is built once and
        // reused while the amount slider moves.
        let flare_base =
            (settings.flare > 0.0).then(|| flare::generate(color_buffer, width, height));

        SpatialBase {
            sharpen_blur,
            texture_base,
            noise_base,
            mask_base,
            wide_blur,
            halation_blur,
            flare_base,
        }
    }

    /// Pass 2 (spatial) that computes its own precomputations.
    /// For one-off calls (export, tests). In the interactive preview
    /// use `run_spatial_pass_with_base` with the cache.
    pub(crate) fn run_spatial_pass(
        settings: &FilterSettings,
        color_buffer: &[u8],
        output: &mut [u8],
        grain_map: &[f32],
        luma_map: &[f32],
        clarity_base: &[f32],
        clarity_sw: usize,
        clarity_sh: usize,
        width: usize,
        height: usize,
    ) {
        let base = Self::compute_spatial_base(settings, color_buffer, luma_map, width, height);
        Self::run_spatial_pass_with_base(
            settings,
            color_buffer,
            output,
            grain_map,
            clarity_base,
            clarity_sw,
            clarity_sh,
            &base,
            width,
            height,
        );
    }

    /// Pass 2 (spatial) on top of the already-prepared `spatial_base` precomputations.
    pub(crate) fn run_spatial_pass_with_base(
        settings: &FilterSettings,
        color_buffer: &[u8],
        output: &mut [u8],
        grain_map: &[f32],
        clarity_base: &[f32],
        clarity_sw: usize,
        clarity_sh: usize,
        spatial_base: &SpatialBase,
        width: usize,
        height: usize,
    ) {
        let s = *settings;

        let n_texture = s.texture / 100.0; // -1.0 .. 1.0
        let n_clarity = s.clarity / 100.0; // -1.0 .. 1.0
        let n_sharpen = s.sharpen / 100.0; // 0.0 .. 1.5
        let n_masking = (s.detail_masking / 100.0).clamp(0.0, 1.0);
        let n_structure = s.structure / 100.0; // -1.0 .. 1.0
        let n_glow = s.glow / 100.0; // 0.0 .. 1.0
        let n_halation = s.halation / 100.0; // 0.0 .. 1.0
        let n_flare = s.flare / 100.0; // 0.0 .. 1.0

        // Per-pixel noise-reduction plan (chroma mix / luma mix + soft thresholds).
        let nr_plan = noise::plan(&s.noise);
        let noise_active = !s.noise.is_identity();

        // Vignette: positional falloff, applied last (after grain).
        let vignette = VignetteParams::new(&s.vignette);
        let vignette_active = !vignette.is_identity();

        // Read from color_buffer (stable color_pass result) — WITHOUT clone!
        let color_snapshot: &[u8] = color_buffer;

        output
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(idx, pixel)| {
                let x = idx % width;
                let y = idx / width;

                let center_idx = idx * 3;
                let mut r = color_snapshot[center_idx] as f32 / 255.0;
                let mut g = color_snapshot[center_idx + 1] as f32 / 255.0;
                let mut b = color_snapshot[center_idx + 2] as f32 / 255.0;

                // ==========================================
                // 🔇 NOISE REDUCTION (before Texture/Clarity/Sharpen)
                // ==========================================
                // The reference is precomputed and edge-aware: chroma is an Oklab
                // a/b low-pass, luma a guided filter. Operating first means the
                // detail stages below work on the cleaned image instead of
                // amplifying the noise.
                if noise_active {
                    if let Some(nb) = &spatial_base.noise_base {
                        let guide = Self::sample_noise_base(nb, x, y, width, height);
                        if let Some(cp) = &nr_plan.chroma {
                            let lin = [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)];
                            let lab = rgb_to_oklab(lin[0], lin[1], lin[2]);
                            let mixed =
                                noise::blend_chroma([lab[1], lab[2]], [guide[0], guide[1]], cp);
                            let out = oklab_to_linear_rgb(lab[0], mixed[0], mixed[1]);
                            r = linear_to_srgb(out[0]).clamp(0.0, 1.0);
                            g = linear_to_srgb(out[1]).clamp(0.0, 1.0);
                            b = linear_to_srgb(out[2]).clamp(0.0, 1.0);
                        }
                        if let Some(lp) = &nr_plan.luma {
                            let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                            let cur_log = y.max(1e-4).log2();
                            let gain = noise::luma_gain(cur_log, guide[2], lp);
                            r = (r * gain).clamp(0.0, 1.0);
                            g = (g * gain).clamp(0.0, 1.0);
                            b = (b * gain).clamp(0.0, 1.0);
                        }
                    }
                }

                // Luma after NR (used by Sharpen below; Clarity recomputes its own).
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                // Shared detail mask for Texture/Clarity/Sharpen: `1` when
                // masking is off, else a soft gate that pulls the detail boosts
                // out of flat/noisy areas while keeping real edges.
                let detail_mask = match &spatial_base.mask_base {
                    Some(mb) if n_masking > 0.0 => {
                        let e = Self::sample_mask_base(mb, x, y, width, height);
                        1.0 + n_masking * (e - 1.0)
                    }
                    _ => 1.0,
                };

                // TEXTURE
                if let Some(tex) = &spatial_base.texture_base {
                    // The base was computed at reduced resolution — sample bilinearly.
                    let blurred = Self::sample_texture_base(tex, x, y, width, height);

                    // Soft blend: original + (original − base) * strength.
                    // Masking tames only the positive (detail-adding) direction;
                    // negative Texture is smoothing and is left alone.
                    let strength = n_texture * 2.0;
                    let strength = if n_texture > 0.0 {
                        strength * detail_mask
                    } else {
                        strength
                    };
                    r = (r + (r - blurred[0]) * strength).clamp(0.0, 1.0);
                    g = (g + (g - blurred[1]) * strength).clamp(0.0, 1.0);
                    b = (b + (b - blurred[2]) * strength).clamp(0.0, 1.0);
                }

                // ---- CLARITY PRO (Logarithmic Local Contrast) ----
                if n_clarity != 0.0 && !clarity_base.is_empty() {
                    // Take the CURRENT luma (including the already applied Texture)
                    // so the gain is computed from the current state.
                    // Cost ~5 flops/pixel — practically free.
                    // We do not recompute the base (`clarity_base`): that is expensive, and the base is
                    // a low-frequency reference.
                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    let scale = Self::clarity_scale(width, height);
                    let fx = x as f32 / scale as f32;
                    let fy = y as f32 / scale as f32;

                    let x0 = (fx as usize).min(clarity_sw - 1);
                    let y0 = (fy as usize).min(clarity_sh - 1);
                    let x1 = (x0 + 1).min(clarity_sw - 1);
                    let y1 = (y0 + 1).min(clarity_sh - 1);
                    let wx = fx - x0 as f32;
                    let wy = fy - y0 as f32;

                    let b00 = clarity_base[y0 * clarity_sw + x0];
                    let b10 = clarity_base[y0 * clarity_sw + x1];
                    let b01 = clarity_base[y1 * clarity_sw + x0];
                    let b11 = clarity_base[y1 * clarity_sw + x1];
                    let base_luma = b00 * (1.0 - wx) * (1.0 - wy)
                        + b10 * wx * (1.0 - wy)
                        + b01 * (1.0 - wx) * wy
                        + b11 * wx * wy;

                    let amount = n_clarity;
                    // Masking tames only positive Clarity (adding local
                    // contrast); negative Clarity is softening and is left alone.
                    let amount = if n_clarity > 0.0 {
                        amount * detail_mask
                    } else {
                        amount
                    };
                    let diff = luma - base_luma;

                    // (1) Soft clipping of the diff
                    let k = 10.0;
                    let safe_diff = diff / (1.0 + diff.abs() * k);

                    // (2) Local-contrast mask (anti-halo)
                    let local_contrast = diff.abs();
                    let edge_mask = 1.0 / (1.0 + local_contrast * 6.0);

                    let punch = 1.0;
                    let mut final_luma = luma + safe_diff * amount * punch * edge_mask;
                    final_luma = final_luma.clamp(0.0, 1.0);

                    let gain = final_luma / luma.max(0.001);
                    r = (r * gain).clamp(0.0, 1.0);
                    g = (g * gain).clamp(0.0, 1.0);
                    b = (b * gain).clamp(0.0, 1.0);
                }

                // ==========================================
                // 🌄 STRUCTURE (broad local contrast)
                // ==========================================
                // Uses the wide low-pass reference (shared with Glow). The
                // larger radius than Clarity gives HDR-like midtone punch
                // instead of micro-contrast.
                if n_structure.abs() > 1e-6 {
                    if let Some(wide) = &spatial_base.wide_blur {
                        let blurred = Self::sample_texture_base(wide, x, y, width, height);
                        let base_luma =
                            0.2126 * blurred[0] + 0.7152 * blurred[1] + 0.0722 * blurred[2];
                        let luma_now = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                        // Masking tames only the positive (contrast-adding)
                        // direction; negative Structure is softening.
                        let amount = if n_structure > 0.0 {
                            n_structure * detail_mask
                        } else {
                            n_structure
                        };
                        if luma_now > 0.001 {
                            let final_luma = creative::apply_structure(luma_now, base_luma, amount);
                            let gain = final_luma / luma_now;
                            r = (r * gain).clamp(0.0, 1.0);
                            g = (g * gain).clamp(0.0, 1.0);
                            b = (b * gain).clamp(0.0, 1.0);
                        }
                    }
                }

                // ==========================================
                // 📸 SHARPENING (Unsharp Mask)
                // ==========================================
                // Sharpen goes last among the details (after Texture/Clarity),
                // as in darktable/Lightroom: local contrast must not
                // amplify sharpening halos. The high-pass is taken from the source luma
                // (`sharpen_blur` is precomputed), and the brightness ratio
                // is applied to the final RGB.
                if let Some(blur) = &spatial_base.sharpen_blur {
                    let high_pass = luma - blur[idx];

                    let max_lighten = 0.20;
                    let max_darken = 0.12;

                    let limited_hp = if high_pass > 0.0 {
                        high_pass.min(max_lighten)
                    } else {
                        high_pass.max(-max_darken)
                    };

                    let strength = n_sharpen * 4.0;
                    let luma_delta = limited_hp * strength * detail_mask;

                    if luma > 0.001 {
                        let ratio = (luma + luma_delta) / luma;
                        r = (r * ratio).clamp(0.0, 1.0);
                        g = (g * ratio).clamp(0.0, 1.0);
                        b = (b * ratio).clamp(0.0, 1.0);
                    }
                }

                // ==========================================
                // ✨ CREATIVE: GLOW / HALATION / LIGHT FLARES
                // ==========================================
                // Additive light from a cached low-pass reference, in the
                // RapidRAW order: Glow, then Halation, then Flares.
                if n_glow > 0.0 {
                    if let Some(wide) = &spatial_base.wide_blur {
                        let blurred = Self::sample_texture_base(wide, x, y, width, height);
                        let out = creative::apply_glow([r, g, b], blurred, n_glow);
                        r = out[0].clamp(0.0, 1.0);
                        g = out[1].clamp(0.0, 1.0);
                        b = out[2].clamp(0.0, 1.0);
                    }
                }
                if n_halation > 0.0 {
                    if let Some(hal) = &spatial_base.halation_blur {
                        let blurred = Self::sample_texture_base(hal, x, y, width, height);
                        let out = creative::apply_halation([r, g, b], blurred, n_halation);
                        r = out[0].clamp(0.0, 1.0);
                        g = out[1].clamp(0.0, 1.0);
                        b = out[2].clamp(0.0, 1.0);
                    }
                }
                if n_flare > 0.0 {
                    if let Some(fb) = &spatial_base.flare_base {
                        let f = fb.sample(x, y, width, height);
                        let out = flare::apply_flare([r, g, b], f, n_flare);
                        r = out[0].clamp(0.0, 1.0);
                        g = out[1].clamp(0.0, 1.0);
                        b = out[2].clamp(0.0, 1.0);
                    }
                }

                // ==========================================
                // 🎞 FILM GRAIN (darktable grain.c: simplex + paper LUT)
                // ==========================================
                if s.grain > 0.0 {
                    // Interleaved map: slot 0 = base, slot 1 = coarser rough
                    // field (present only when roughness was engaged).
                    if let Some(&base) = grain_map.get(idx * 2) {
                        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                        let noise = match grain_map.get(idx * 2 + 1) {
                            Some(&rough) => crate::pipeline::grain::blend_noise(
                                base,
                                rough,
                                s.grain_roughness / 100.0,
                            ),
                            None => base,
                        };
                        let delta =
                            crate::pipeline::grain::grain_delta(noise, s.grain / 100.0, luma);
                        // Monochrome lightness grain: the same shift in all channels.
                        r = (r + delta).clamp(0.0, 1.0);
                        g = (g + delta).clamp(0.0, 1.0);
                        b = (b + delta).clamp(0.0, 1.0);
                    }
                }

                // ==========================================
                // 🌑 VIGNETTE (positional, after grain)
                // ==========================================
                if vignette_active {
                    let mut rgb = [r, g, b];
                    vignette.apply(&mut rgb, x, y, width, height);
                    r = rgb[0];
                    g = rgb[1];
                    b = rgb[2];
                }

                pixel[0] = (r * 255.0).clamp(0.0, 255.0) as u8;
                pixel[1] = (g * 255.0).clamp(0.0, 255.0) as u8;
                pixel[2] = (b * 255.0).clamp(0.0, 255.0) as u8;
            });
    }

    // ==========================================
    // PHASE 3: STAGED SPATIAL PIPELINE
    //
    // The interactive drag tier splits the spatial pass into a *detail head*
    // (noise reduction → Texture / Clarity / Structure / Sharpen) and an
    // *effects tail* (Glow / Halation / Flare / Grain / Vignette). The head is
    // cached in f32 so dragging an effects slider can skip the expensive detail
    // work, and the tail reads the f32 head directly (no intermediate 8-bit
    // round-trip). The two halves together are byte-identical to the monolithic
    // `run_spatial_pass_with_base`; the `spatial_stages_match_monolithic` test
    // guards that (keep the per-pixel math here in sync with the loop above).
    // ==========================================

    /// Detail head of the spatial pass for one pixel: noise reduction, then
    /// Texture / Clarity / Structure / Sharpen. Returns display-referred f32 RGB.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn spatial_detail_pixel(
        rgb: [f32; 3],
        x: usize,
        y: usize,
        idx: usize,
        base: &SpatialBase,
        params: &SpatialParams,
        clarity_base: &[f32],
        clarity_sw: usize,
        clarity_sh: usize,
        width: usize,
        height: usize,
    ) -> [f32; 3] {
        let (mut r, mut g, mut b) = (rgb[0], rgb[1], rgb[2]);

        // 🔇 NOISE REDUCTION (before Texture/Clarity/Sharpen)
        if params.noise_active {
            if let Some(nb) = &base.noise_base {
                let guide = Self::sample_noise_base(nb, x, y, width, height);
                if let Some(cp) = &params.nr_plan.chroma {
                    let lin = [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)];
                    let lab = rgb_to_oklab(lin[0], lin[1], lin[2]);
                    let mixed = noise::blend_chroma([lab[1], lab[2]], [guide[0], guide[1]], cp);
                    let out = oklab_to_linear_rgb(lab[0], mixed[0], mixed[1]);
                    r = linear_to_srgb(out[0]).clamp(0.0, 1.0);
                    g = linear_to_srgb(out[1]).clamp(0.0, 1.0);
                    b = linear_to_srgb(out[2]).clamp(0.0, 1.0);
                }
                if let Some(lp) = &params.nr_plan.luma {
                    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    let cur_log = y.max(1e-4).log2();
                    let gain = noise::luma_gain(cur_log, guide[2], lp);
                    r = (r * gain).clamp(0.0, 1.0);
                    g = (g * gain).clamp(0.0, 1.0);
                    b = (b * gain).clamp(0.0, 1.0);
                }
            }
        }

        // Luma after NR (used by Sharpen below; Clarity recomputes its own).
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        // Shared detail mask for Texture/Clarity/Sharpen.
        let detail_mask = match &base.mask_base {
            Some(mb) if params.n_masking > 0.0 => {
                let e = Self::sample_mask_base(mb, x, y, width, height);
                1.0 + params.n_masking * (e - 1.0)
            }
            _ => 1.0,
        };

        // TEXTURE
        if let Some(tex) = &base.texture_base {
            let blurred = Self::sample_texture_base(tex, x, y, width, height);
            let strength = params.n_texture * 2.0;
            let strength = if params.n_texture > 0.0 {
                strength * detail_mask
            } else {
                strength
            };
            r = (r + (r - blurred[0]) * strength).clamp(0.0, 1.0);
            g = (g + (g - blurred[1]) * strength).clamp(0.0, 1.0);
            b = (b + (b - blurred[2]) * strength).clamp(0.0, 1.0);
        }

        // ---- CLARITY PRO (Logarithmic Local Contrast) ----
        if params.n_clarity != 0.0 && !clarity_base.is_empty() {
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let scale = Self::clarity_scale(width, height);
            let fx = x as f32 / scale as f32;
            let fy = y as f32 / scale as f32;

            let x0 = (fx as usize).min(clarity_sw - 1);
            let y0 = (fy as usize).min(clarity_sh - 1);
            let x1 = (x0 + 1).min(clarity_sw - 1);
            let y1 = (y0 + 1).min(clarity_sh - 1);
            let wx = fx - x0 as f32;
            let wy = fy - y0 as f32;

            let b00 = clarity_base[y0 * clarity_sw + x0];
            let b10 = clarity_base[y0 * clarity_sw + x1];
            let b01 = clarity_base[y1 * clarity_sw + x0];
            let b11 = clarity_base[y1 * clarity_sw + x1];
            let base_luma = b00 * (1.0 - wx) * (1.0 - wy)
                + b10 * wx * (1.0 - wy)
                + b01 * (1.0 - wx) * wy
                + b11 * wx * wy;

            let amount = params.n_clarity;
            let amount = if params.n_clarity > 0.0 {
                amount * detail_mask
            } else {
                amount
            };
            let diff = luma - base_luma;

            let k = 10.0;
            let safe_diff = diff / (1.0 + diff.abs() * k);

            let local_contrast = diff.abs();
            let edge_mask = 1.0 / (1.0 + local_contrast * 6.0);

            let punch = 1.0;
            let mut final_luma = luma + safe_diff * amount * punch * edge_mask;
            final_luma = final_luma.clamp(0.0, 1.0);

            let gain = final_luma / luma.max(0.001);
            r = (r * gain).clamp(0.0, 1.0);
            g = (g * gain).clamp(0.0, 1.0);
            b = (b * gain).clamp(0.0, 1.0);
        }

        // ==========================================
        // 🌄 STRUCTURE (broad local contrast)
        // ==========================================
        if params.n_structure.abs() > 1e-6 {
            if let Some(wide) = &base.wide_blur {
                let blurred = Self::sample_texture_base(wide, x, y, width, height);
                let base_luma = 0.2126 * blurred[0] + 0.7152 * blurred[1] + 0.0722 * blurred[2];
                let luma_now = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let amount = if params.n_structure > 0.0 {
                    params.n_structure * detail_mask
                } else {
                    params.n_structure
                };
                if luma_now > 0.001 {
                    let final_luma = creative::apply_structure(luma_now, base_luma, amount);
                    let gain = final_luma / luma_now;
                    r = (r * gain).clamp(0.0, 1.0);
                    g = (g * gain).clamp(0.0, 1.0);
                    b = (b * gain).clamp(0.0, 1.0);
                }
            }
        }

        // ==========================================
        // 📸 SHARPENING (Unsharp Mask)
        // ==========================================
        if let Some(blur) = &base.sharpen_blur {
            let high_pass = luma - blur[idx];

            let max_lighten = 0.20;
            let max_darken = 0.12;

            let limited_hp = if high_pass > 0.0 {
                high_pass.min(max_lighten)
            } else {
                high_pass.max(-max_darken)
            };

            let strength = params.n_sharpen * 4.0;
            let luma_delta = limited_hp * strength * detail_mask;

            if luma > 0.001 {
                let ratio = (luma + luma_delta) / luma;
                r = (r * ratio).clamp(0.0, 1.0);
                g = (g * ratio).clamp(0.0, 1.0);
                b = (b * ratio).clamp(0.0, 1.0);
            }
        }

        [r, g, b]
    }

    /// Effects tail of the spatial pass for one pixel: Glow / Halation / Flare /
    /// Grain / Vignette, reading the detail head's display-referred f32 RGB.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn spatial_effects_pixel(
        rgb: [f32; 3],
        x: usize,
        y: usize,
        idx: usize,
        grain_map: &[f32],
        base: &SpatialBase,
        params: &SpatialParams,
        width: usize,
        height: usize,
    ) -> [f32; 3] {
        let (mut r, mut g, mut b) = (rgb[0], rgb[1], rgb[2]);

        // ✨ CREATIVE: GLOW / HALATION / LIGHT FLARES
        if params.n_glow > 0.0 {
            if let Some(wide) = &base.wide_blur {
                let blurred = Self::sample_texture_base(wide, x, y, width, height);
                let out = creative::apply_glow([r, g, b], blurred, params.n_glow);
                r = out[0].clamp(0.0, 1.0);
                g = out[1].clamp(0.0, 1.0);
                b = out[2].clamp(0.0, 1.0);
            }
        }
        if params.n_halation > 0.0 {
            if let Some(hal) = &base.halation_blur {
                let blurred = Self::sample_texture_base(hal, x, y, width, height);
                let out = creative::apply_halation([r, g, b], blurred, params.n_halation);
                r = out[0].clamp(0.0, 1.0);
                g = out[1].clamp(0.0, 1.0);
                b = out[2].clamp(0.0, 1.0);
            }
        }
        if params.n_flare > 0.0 {
            if let Some(fb) = &base.flare_base {
                let f = fb.sample(x, y, width, height);
                let out = flare::apply_flare([r, g, b], f, params.n_flare);
                r = out[0].clamp(0.0, 1.0);
                g = out[1].clamp(0.0, 1.0);
                b = out[2].clamp(0.0, 1.0);
            }
        }

        // 🎞 FILM GRAIN
        if params.grain_amount > 0.0 {
            if let Some(&gbase) = grain_map.get(idx * 2) {
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let noise = match grain_map.get(idx * 2 + 1) {
                    Some(&rough) => crate::pipeline::grain::blend_noise(
                        gbase,
                        rough,
                        params.grain_roughness / 100.0,
                    ),
                    None => gbase,
                };
                let delta = crate::pipeline::grain::grain_delta(
                    noise,
                    params.grain_amount / 100.0,
                    luma,
                );
                r = (r + delta).clamp(0.0, 1.0);
                g = (g + delta).clamp(0.0, 1.0);
                b = (b + delta).clamp(0.0, 1.0);
            }
        }

        // 🌑 VIGNETTE (positional, after grain)
        if params.vignette_active {
            let mut c = [r, g, b];
            params.vignette.apply(&mut c, x, y, width, height);
            r = c[0];
            g = c[1];
            b = c[2];
        }

        [r, g, b]
    }

    /// Detail head: runs [`TinyLumaApp::spatial_detail_pixel`] over the
    /// whole frame and writes display-referred f32 RGB into `out`. Kept in f32 so
    /// the effects tail is byte-identical to the monolithic pass.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn spatial_detail_pass(
        settings: &FilterSettings,
        input: &[u8],
        clarity_base: &[f32],
        clarity_sw: usize,
        clarity_sh: usize,
        spatial_base: &SpatialBase,
        width: usize,
        height: usize,
        out: &mut [f32],
    ) {
        let params = SpatialParams::new(settings);
        out.par_chunks_mut(3).enumerate().for_each(|(idx, px)| {
            let x = idx % width;
            let y = idx / width;
            let ci = idx * 3;
            let d = Self::spatial_detail_pixel(
                [
                    input[ci] as f32 / 255.0,
                    input[ci + 1] as f32 / 255.0,
                    input[ci + 2] as f32 / 255.0,
                ],
                x,
                y,
                idx,
                spatial_base,
                &params,
                clarity_base,
                clarity_sw,
                clarity_sh,
                width,
                height,
            );
            px[0] = d[0];
            px[1] = d[1];
            px[2] = d[2];
        });
    }

    /// Effects tail: runs [`TinyLumaApp::spatial_effects_pixel`] over the
    /// detail head and writes the final u8 RGB into `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn spatial_effects_pass(
        settings: &FilterSettings,
        detail: &[f32],
        grain_map: &[f32],
        spatial_base: &SpatialBase,
        width: usize,
        height: usize,
        out: &mut [u8],
    ) {
        let params = SpatialParams::new(settings);
        out.par_chunks_mut(3).enumerate().for_each(|(idx, px)| {
            let x = idx % width;
            let y = idx / width;
            let ci = idx * 3;
            let e = Self::spatial_effects_pixel(
                [detail[ci], detail[ci + 1], detail[ci + 2]],
                x,
                y,
                idx,
                grain_map,
                spatial_base,
                &params,
                width,
                height,
            );
            px[0] = (e[0] * 255.0).clamp(0.0, 255.0) as u8;
            px[1] = (e[1] * 255.0).clamp(0.0, 255.0) as u8;
            px[2] = (e[2] * 255.0).clamp(0.0, 255.0) as u8;
        });
    }

    // ==========================================
    // PHASE 3: REDUCED-RESOLUTION INTERACTIVE DRAG TIER
    // ==========================================

    /// Dimensions of the reduced drag buffer for a `width × height` preview
    /// (long side capped at [`DRAG_WORK_MAX`], aspect ratio preserved).
    pub(crate) fn drag_dims(width: usize, height: usize) -> (usize, usize) {
        let m = width.max(height);
        if m <= DRAG_WORK_MAX {
            return (width, height);
        }
        let scale = DRAG_WORK_MAX as f32 / m as f32;
        (
            ((width as f32 * scale).round() as usize).max(1),
            ((height as f32 * scale).round() as usize).max(1),
        )
    }

    // ==========================================
    // PHASE 4: DYNAMIC VIEWPORT PROXY
    // ==========================================

    /// Long-side target of the interactive proxy for a viewport.
    ///
    /// `viewport_px` is the viewport's long side in logical points; the on-screen
    /// image spans roughly `viewport_px · zoom · pixels_per_point` physical pixels
    /// when fitted, so that product sizes the proxy. The value is snapped up to
    /// the next [`PROXY_LADDER`] rung (never below [`PROXY_WORK_MIN`]) so the
    /// proxy is rebuilt at most once per rung. Not yet clamped to the source —
    /// see [`Self::proxy_dims`].
    pub(crate) fn proxy_target_long(viewport_px: f32, zoom: f32, ppp: f32) -> usize {
        let desired = (viewport_px * zoom * ppp).ceil().max(0.0) as usize;
        let desired = desired.max(PROXY_WORK_MIN);
        PROXY_LADDER
            .iter()
            .copied()
            .find(|&r| r >= desired)
            .unwrap_or(*PROXY_LADDER.last().unwrap())
    }

    /// Dimensions of the interactive proxy for a source frame scaled so
    /// its long side is `target_long`. Aspect ratio is preserved, the result is
    /// never upscaled beyond the source, and both axes stay ≥ 1.
    pub(crate) fn proxy_dims_for_long(
        source_w: usize,
        source_h: usize,
        target_long: usize,
    ) -> (usize, usize) {
        let sw = source_w.max(1);
        let sh = source_h.max(1);
        let source_long = sw.max(sh);
        if target_long >= source_long {
            return (sw, sh);
        }
        let target = target_long.max(1);
        let scale = target as f32 / source_long as f32;
        (
            ((sw as f32 * scale).round() as usize).max(1),
            ((sh as f32 * scale).round() as usize).max(1),
        )
    }

    /// Dimensions of the interactive working proxy for a source frame at
    /// the given viewport / zoom / pixel density. Composes
    /// [`Self::proxy_target_long`] with [`Self::proxy_dims_for_long`]; the proxy
    /// never exceeds the source and never falls below 1 px on either axis.
    pub(crate) fn proxy_dims(
        source_w: usize,
        source_h: usize,
        viewport_px: f32,
        zoom: f32,
        ppp: f32,
    ) -> (usize, usize) {
        let source_long = source_w.max(source_h).max(1);
        let target = Self::proxy_target_long(viewport_px, zoom, ppp).min(source_long);
        Self::proxy_dims_for_long(source_w, source_h, target)
    }

    /// Re-keys the resolution-implicit pipeline caches after the working
    /// proxy (`preview_base`) changed size. `new_color_len` is the new working
    /// image length (`width * height * 3`): the cached colour buffer is re-fitted
    /// to it, so it can no longer be observed at the previous resolution's length.
    /// Only clears the validity flags, bumps the input revision and resets the
    /// upload keys — it does NOT force a re-render, because the background worker
    /// already produced the display pixels (see [`Self::apply_proxy_reply`]); the
    /// next settings change at the new resolution rebuilds them.
    ///
    /// Defence in depth, not the load-bearing fix: [`Self::process_preview`]
    /// re-fits the colour buffer at its copy site as well, because a new image or
    /// a close changes the working resolution without going through this helper.
    fn invalidate_resolution_caches(&mut self, new_color_len: usize) {
        // A resolution swap must not leave a cached buffer sized for the previous
        // resolution: `copy_from_slice` panics on a length mismatch, and
        // `panic = "abort"` turns that into an instant process kill.
        if self.color_buffer.len() != new_color_len {
            self.color_buffer.resize(new_color_len, 0);
        }
        self.color_buffer_valid = false;
        self.luma_cache_valid = false;
        self.clarity_cache_valid = false;
        self.spatial_base_valid = false;
        self.dehaze_cache_valid = false;
        self.drag_tier.tier_valid = false;
        // The spatial input resolution changed — invalidate the drag-tier caches.
        self.spatial_input_revision = self.spatial_input_revision.wrapping_add(1);
        // Force a fresh texture upload at the new size (the upload key has no size).
        self.upload_display_key = None;
        self.upload_orig_key = None;
    }

    /// Derives the desired proxy resolution from the viewport
    /// (`canvas_long · zoom · ppp`) and records when it changed, for the rebuild
    /// debounce. No-op until a frame is open. Sizes against the higher-res source
    /// once it has arrived, else against the current base (so the target is
    /// defined and never exceeds what is available).
    pub(crate) fn update_proxy_target(&mut self, ctx: &egui::Context, canvas_long: f32, ppp: f32) {
        if self.preview_base.is_none() || self.retouch.active {
            return;
        }
        let (sw, sh) = match self.proxy_source_dims() {
            Some(d) => d,
            None => match self.preview_base.as_ref() {
                Some(b) => (b.width() as usize, b.height() as usize),
                None => return,
            },
        };
        if sw == 0 || sh == 0 {
            return;
        }
        let dims = Self::proxy_dims(sw, sh, canvas_long, self.zoom_scale, ppp);
        if dims != self.desired_proxy {
            self.desired_proxy = dims;
            self.proxy_change_time = Instant::now();
            // Keep frames coming so the debounce fires even without further input.
            ctx.request_repaint_after(PROXY_SETTLE);
        }
    }

    /// Enqueues a background rebuild of the working proxy from the
    /// higher-res source once the desired resolution has settled (or a rebuild was
    /// explicitly requested). The previous proxy stays visible (stretched) while
    /// the worker downscales, heals and renders; [`Self::poll_proxy_render`]
    /// installs the reply. No-op when the desired resolution already matches.
    fn maybe_rebuild_proxy(&mut self, ctx: &egui::Context) {
        if self.preview_base.is_none() || self.retouch.active {
            return;
        }
        if self.desired_proxy == self.proxy_dims && !self.proxy_rebuild_now {
            return;
        }
        if self.desired_proxy.0 == 0 || self.desired_proxy.1 == 0 {
            return;
        }
        // A job for this exact target is already in flight — coalesce.
        if self.proxy_pending_target == Some(self.desired_proxy) {
            return;
        }
        // Debounce: while a zoom/size gesture is still settling, keep showing the
        // previous (stretched) proxy and wait.
        if !self.proxy_rebuild_now {
            let elapsed = self.proxy_change_time.elapsed();
            if elapsed < PROXY_SETTLE {
                ctx.request_repaint_after(PROXY_SETTLE - elapsed);
                return;
            }
        }
        // A higher-res source is required to derive a new resolution.
        let Some(source) = self.proxy_source.clone() else {
            self.proxy_rebuild_now = false;
            return;
        };
        let (sw, sh) = (source.width() as usize, source.height() as usize);
        // Never upscale past the source.
        let (dw, dh) = self.desired_proxy;
        let (dw, dh) = if dw > sw || dh > sh { (sw, sh) } else { (dw, dh) };
        if dw == 0 || dh == 0 {
            self.proxy_rebuild_now = false;
            return;
        }
        self.proxy_job_revision = self.proxy_job_revision.wrapping_add(1);
        let revision = self.proxy_job_revision;
        let req = ProxyRequest {
            revision,
            source,
            target: (dw, dh),
            settings: self.settings,
            lut: self.active_lut.clone(),
            layer: Arc::new(self.retouch.layer.clone()),
        };
        self.proxy_pending_target = Some((dw, dh));
        self.proxy_rebuild_now = false;
        self.proxy_worker_mut(ctx).enqueue(req);
        // Keep frames coming so the reply is applied promptly.
        ctx.request_repaint();
    }

    /// Lazily spawns the background proxy-render worker.
    fn proxy_worker_mut(&mut self, ctx: &egui::Context) -> &mut super::proxy_job::ProxyWorker {
        if self.proxy_worker.is_none() {
            self.proxy_worker = Some(super::proxy_job::ProxyWorker::spawn(ctx));
        }
        self.proxy_worker
            .as_mut()
            .expect("worker spawned just above")
    }

    /// Applies finished proxy-render jobs. Called once per frame on the
    /// UI thread; never blocks. Stale replies are discarded by revision.
    pub(crate) fn poll_proxy_render(&mut self, ctx: &egui::Context) {
        if let Some(w) = &mut self.proxy_worker {
            w.pump();
        }
        let mut replies = Vec::new();
        if let Some(w) = &self.proxy_worker {
            while let Ok(reply) = w.rx.try_recv() {
                replies.push(reply);
            }
        }
        for reply in replies {
            self.apply_proxy_reply(reply, ctx);
        }
    }

    /// Installs a worker-rendered proxy as the new working base and
    /// shows its pixels, re-keying the resolution-implicit caches **without**
    /// forcing a re-render (the worker already produced the display pixels).
    fn apply_proxy_reply(&mut self, reply: ProxyReply, ctx: &egui::Context) {
        match reply {
            ProxyReply::Rendered {
                revision,
                proxy,
                processed,
                dims,
            } => {
                // Superseded (a newer target was requested) — discard.
                if revision != self.proxy_job_revision {
                    return;
                }
                // A retouch session owns the canvas; the proxy is not shown.
                if self.retouch.active {
                    self.proxy_pending_target = None;
                    return;
                }
                self.proxy_pending_target = None;
                self.preview_base = Some(proxy);
                self.processed_pixels = processed;
                self.proxy_dims = dims;
                self.desired_proxy = dims;
                self.proxy_rebuild_now = false;
                // The base changed size: re-key the caches for the new resolution
                // but do NOT mark the pipeline dirty (the worker already rendered).
                self.invalidate_resolution_caches(self.processed_pixels.len());
                self.processed_revision = self.processed_revision.wrapping_add(1);
                self.upload_preview_textures(ctx);
                ctx.request_repaint();
            }
        }
    }

    /// Cancels any in-flight proxy-render job (frame switch, close,
    /// retouch entry). The worker drops a superseded job before doing the work.
    pub(crate) fn invalidate_proxy_render(&mut self) {
        if let Some(w) = &mut self.proxy_worker {
            w.invalidate();
        }
        self.proxy_job_revision = self.proxy_job_revision.wrapping_add(1);
        self.proxy_pending_target = None;
    }

    /// Box-downscales an interleaved RGB8 buffer (area average per output pixel).
    pub(crate) fn downscale_rgb(src: &[u8], w: usize, h: usize, dw: usize, dh: usize) -> Vec<u8> {
        let mut out = vec![0u8; dw * dh * 3];
        if dw == w && dh == h {
            let n = out.len();
            out.copy_from_slice(&src[..n]);
            return out;
        }
        let sx = w as f32 / dw as f32;
        let sy = h as f32 / dh as f32;
        out.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
            let dx = i % dw;
            let dy = i / dw;
            let x0 = (dx as f32 * sx) as usize;
            let x1 = (((dx + 1) as f32 * sx) as usize).clamp(x0 + 1, w);
            let y0 = (dy as f32 * sy) as usize;
            let y1 = (((dy + 1) as f32 * sy) as usize).clamp(y0 + 1, h);
            let mut acc = [0.0f32; 3];
            let mut cnt = 0.0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    let b = (y * w + x) * 3;
                    acc[0] += src[b] as f32;
                    acc[1] += src[b + 1] as f32;
                    acc[2] += src[b + 2] as f32;
                    cnt += 1.0;
                }
            }
            let inv = 1.0 / cnt.max(1.0);
            o[0] = (acc[0] * inv) as u8;
            o[1] = (acc[1] * inv) as u8;
            o[2] = (acc[2] * inv) as u8;
        });
        out
    }

    /// Signature of the settings that affect *construction* of the spatial bases
    /// (which effects are engaged) plus the drag resolution.
    fn drag_base_signature(s: &FilterSettings, dw: usize, dh: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        dw.hash(&mut h);
        dh.hash(&mut h);
        (s.texture != 0.0).hash(&mut h);
        (s.sharpen > 0.0).hash(&mut h);
        (!s.noise.is_identity()).hash(&mut h);
        (s.detail_masking > 0.0).hash(&mut h);
        (s.glow > 0.0).hash(&mut h);
        (s.structure != 0.0).hash(&mut h);
        (s.halation > 0.0).hash(&mut h);
        (s.flare > 0.0).hash(&mut h);
        h.finish()
    }

    /// Signature of the settings the detail head depends on.
    fn drag_detail_signature(s: &FilterSettings) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for v in [
            s.texture,
            s.clarity,
            s.structure,
            s.sharpen,
            s.detail_masking,
            s.noise.luma,
            s.noise.chroma,
            s.noise.detail,
        ] {
            v.to_bits().hash(&mut h);
        }
        h.finish()
    }

    /// Renders the spatial pass at the reduced drag resolution and
    /// uploads the smaller texture. The cached input/bases and the f32 detail
    /// head are reused when only effects changed, so dragging an effects slider
    /// skips the detail work. Returns the drag dimensions actually used.
    fn render_spatial_drag_tier(
        &mut self,
        ctx: &egui::Context,
        width: usize,
        height: usize,
        use_dehaze: bool,
    ) -> (usize, usize) {
        let (dw, dh) = Self::drag_dims(width, height);
        let settings = self.settings;

        // Rebuild the downscaled input + precomputations only when the input
        // revision, the resolution or the base-affecting settings changed.
        let base_sig = Self::drag_base_signature(&settings, dw, dh);
        let tier_rev = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            self.spatial_input_revision.hash(&mut h);
            base_sig.hash(&mut h);
            h.finish()
        };

        if !self.drag_tier.tier_valid || self.drag_tier.tier_rev != tier_rev {
            let small = {
                let src: &[u8] = if use_dehaze {
                    &self.dehazed_buffer
                } else {
                    &self.color_buffer
                };
                Self::downscale_rgb(src, width, height, dw, dh)
            };
            let luma = Self::compute_luma_map(&small);
            let (clarity, csw, csh) = if settings.clarity != 0.0 {
                Self::compute_clarity_cache(&luma, dw, dh)
            } else {
                (Vec::new(), 0, 0)
            };
            let spatial_base = Self::compute_spatial_base(&settings, &small, &luma, dw, dh);

            let tier = &mut self.drag_tier;
            tier.input = small;
            tier.clarity = clarity;
            tier.clarity_sw = csw;
            tier.clarity_sh = csh;
            tier.spatial_base = spatial_base;
            tier.detail_valid = false;
            tier.tier_rev = tier_rev;
            tier.tier_valid = true;
        }

        // Detail head: skipped entirely when only the effects changed.
        let detail_sig = Self::drag_detail_signature(&settings);
        if !self.drag_tier.detail_valid || self.drag_tier.detail_sig != detail_sig {
            let n = dw * dh * 3;
            if self.drag_tier.detail.len() != n {
                self.drag_tier.detail.resize(n, 0.0);
            }
            Self::spatial_detail_pass(
                &settings,
                &self.drag_tier.input,
                &self.drag_tier.clarity,
                self.drag_tier.clarity_sw,
                self.drag_tier.clarity_sh,
                &self.drag_tier.spatial_base,
                dw,
                dh,
                &mut self.drag_tier.detail,
            );
            self.drag_tier.detail_sig = detail_sig;
            self.drag_tier.detail_valid = true;
        }

        // Grain map at the drag resolution.
        let grain_key = (settings.grain_size.to_bits(), settings.grain_roughness > 0.0);
        if !self.drag_tier.grain_valid
            || self.drag_tier.grain_dims != (dw, dh)
            || self.drag_tier.grain_key != grain_key
        {
            let g = if settings.grain != 0.0 {
                Self::generate_grain_map(
                    dw,
                    dh,
                    42,
                    settings.grain_size,
                    settings.grain_roughness > 0.0,
                )
            } else {
                Vec::new()
            };
            self.drag_tier.grain = g;
            self.drag_tier.grain_dims = (dw, dh);
            self.drag_tier.grain_key = grain_key;
            self.drag_tier.grain_valid = true;
        }

        // Effects tail: always runs (cheap) on top of the cached detail head.
        let n = dw * dh * 3;
        if self.drag_tier.output.len() != n {
            self.drag_tier.output.resize(n, 0);
        }
        Self::spatial_effects_pass(
            &settings,
            &self.drag_tier.detail,
            &self.drag_tier.grain,
            &self.drag_tier.spatial_base,
            dw,
            dh,
            &mut self.drag_tier.output,
        );

        // Upload the reduced texture (stretched to the same on-screen size — the
        // canvas derives its scale from the viewport, not the texture size).
        self.processed_revision = self.processed_revision.wrapping_add(1);
        let overlay = self.clipping.overlay();
        let data: &[u8] = if overlay.any() {
            let mut buf = std::mem::take(&mut self.scratch_clip);
            buf.clear();
            buf.extend_from_slice(&self.drag_tier.output);
            overlay.paint(&mut buf);
            self.scratch_clip = buf;
            &self.scratch_clip
        } else {
            &self.drag_tier.output
        };
        let img = egui::ColorImage::from_rgb([dw, dh], data);
        if let Some(tex) = &mut self.texture {
            tex.set(img, egui::TextureOptions::LINEAR);
        } else {
            self.texture = Some(ctx.load_texture("preview", img, egui::TextureOptions::LINEAR));
        }
        // Force a fresh full-resolution upload when the drag ends.
        self.upload_display_key = None;
        self.histogram = super::histogram::Histogram::of_rgb8(&self.drag_tier.output);
        self.histogram_revision = self.processed_revision;
        self.clip_overlay_applied = overlay;
        (dw, dh)
    }

    // ==========================================
    // SAVING

    pub(crate) fn bake_temp_lut(
        settings: &FilterSettings,
        lut_path: &Option<PathBuf>,
    ) -> Option<Lut3D> {
        // Load the LUT from a file if a path is given.
        let active_lut = lut_path
            .as_ref()
            .and_then(|path| Lut3D::load_from_file(path));
        Some(Self::bake_lut(settings, active_lut.as_ref(), 64))
    }

    // ==========================================
    // SHARED HELPERS FOR PREVIEW AND EXPORT
    // ==========================================

    /// Scale of the detail group relative to the reference (preview cap).
    ///
    /// `1.0` for frames ≤ `DETAIL_WORK_MAX` on the long side, then grows
    /// linearly.
    fn detail_scale(width: usize, height: usize) -> f32 {
        (width.max(height) as f32 / DETAIL_WORK_MAX as f32).max(1.0)
    }

    /// Sub-reference radius factor for the detail group.
    ///
    /// The detail kernels (the Texture/Clarity bilateral radius, the noise
    /// radii, the mask radius, the Sharpen radius) are authored at the 1200 px
    /// reference grid — see [`Self::detail_scale`], which downsamples the
    /// working grid to ~1200 for inputs at or above the reference. On a proxy
    /// *below* the reference (ladder rungs 640/960) that fixed radius is
    /// relatively too large, so it is shrunk here by the same ratio; at or above
    /// the reference this is exactly `1.0` and the behaviour is unchanged.
    fn detail_radius_factor(width: usize, height: usize) -> f32 {
        (width.max(height) as f32 / DETAIL_WORK_MAX as f32).min(1.0)
    }

    /// Downsample factor for Clarity taking resolution into account.
    fn clarity_scale(width: usize, height: usize) -> usize {
        (CLARITY_SCALE as f32 * Self::detail_scale(width, height))
            .round()
            .max(1.0) as usize
    }

    /// Normalized 1D Gaussian kernel, half-width `radius`, σ = 0.85·radius.
    /// With `radius = 1` this is the classic 1-2-1 kernel.
    fn gaussian_kernel(radius: usize) -> Vec<f32> {
        let radius = radius.max(1);
        let sigma = radius as f32 * 0.85;
        let mut kernel = vec![0.0f32; 2 * radius + 1];
        let mut ksum = 0.0f32;
        for (i, k) in kernel.iter_mut().enumerate() {
            let d = i as f32 - radius as f32;
            *k = (-(d * d) / (2.0 * sigma * sigma)).exp();
            ksum += *k;
        }
        for k in &mut kernel {
            *k /= ksum;
        }
        kernel
    }

    /// Separable convolution of a single-channel map with a 1D `kernel`
    /// (its half-width is `(kernel.len() - 1) / 2`).
    fn convolve_separable(src: &[f32], w: usize, h: usize, kernel: &[f32]) -> Vec<f32> {
        let radius = (kernel.len() - 1) / 2;
        let mut tmp = vec![0.0f32; w * h];
        tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let base = y * w;
            for (x, out) in row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (k, &kv) in kernel.iter().enumerate() {
                    let sx = (x as isize + k as isize - radius as isize).clamp(0, w as isize - 1)
                        as usize;
                    acc += src[base + sx] * kv;
                }
                *out = acc;
            }
        });

        let mut out = vec![0.0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (k, &kv) in kernel.iter().enumerate() {
                    let sy = (y as isize + k as isize - radius as isize).clamp(0, h as isize - 1)
                        as usize;
                    acc += tmp[sy * w + x] * kv;
                }
                *o = acc;
            }
        });
        out
    }

    /// Separable Gaussian blur of a single-channel map with a **fractional**
    /// radius that scales with resolution.
    ///
    /// At integer radii the weights are exactly the legacy kernel (`radius = 1`
    /// → 1-2-1). Between integers the two neighbouring integer kernels are
    /// blended *before* the convolution (valid because convolution is linear),
    /// so the blur grows smoothly instead of in integer steps — this removed the
    /// weak-sharpen dip in the 1.2–1.8 MP range — while still costing a single
    /// separable pass.
    fn gaussian_blur_map(src: &[f32], w: usize, h: usize, radius: f32) -> Vec<f32> {
        let r0 = radius.floor().max(1.0) as usize;
        let frac = (radius - r0 as f32).clamp(0.0, 1.0);
        if frac < 1e-4 {
            return Self::convolve_separable(src, w, h, &Self::gaussian_kernel(r0));
        }
        let k0 = Self::gaussian_kernel(r0);
        let k1 = Self::gaussian_kernel(r0 + 1);
        let half = r0 + 1;
        let mut kernel = vec![0.0f32; 2 * half + 1];
        for (i, k) in kernel.iter_mut().enumerate() {
            let d = i as isize - half as isize;
            let v0 = if d.unsigned_abs() <= r0 {
                k0[(d + r0 as isize) as usize]
            } else {
                0.0
            };
            *k = v0 * (1.0 - frac) + k1[i] * frac;
        }
        Self::convolve_separable(src, w, h, &kernel)
    }

    /// Bilinear sampling of the RGB texture base.
    fn sample_texture_base(base: &TextureBase, x: usize, y: usize, w: usize, h: usize) -> [f32; 3] {
        let (sw, sh) = (base.sw, base.sh);
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
            let v00 = base.data[(ya * sw + xa) * 3 + c];
            let v10 = base.data[(ya * sw + xb) * 3 + c];
            let v01 = base.data[(yb * sw + xa) * 3 + c];
            let v11 = base.data[(yb * sw + xb) * 3 + c];
            let top = v00 * (1.0 - tx) + v10 * tx;
            let bot = v01 * (1.0 - tx) + v11 * tx;
            out[c] = top * (1.0 - ty) + bot * ty;
        }
        out
    }

    /// Texture base: bilateral blur at a reduced (relative to
    /// fixed) resolution.
    ///
    /// This way the effect is the same in preview and export, and the cost does not grow with
    /// resolution. With `scale = 1` (preview) it is exactly the former full-res
    /// bilateral filter of radius 2.
    fn compute_texture_base(
        color: &[u8],
        luma: &[f32],
        w: usize,
        h: usize,
        scale: f32,
    ) -> TextureBase {
        let sw = ((w as f32 / scale).round() as usize).max(1);
        let sh = ((h as f32 / scale).round() as usize).max(1);

        // Box-downsample luma and RGB into the working resolution.
        let sx = w as f32 / sw as f32;
        let sy = h as f32 / sh as f32;
        let mut small_luma = vec![0.0f32; sw * sh];
        let mut small_rgb = vec![0.0f32; sw * sh * 3];
        small_luma
            .par_iter_mut()
            .zip(small_rgb.par_chunks_mut(3))
            .enumerate()
            .for_each(|(i, (lo, rgb))| {
                let dx = i % sw;
                let dy = i / sw;
                let x0 = (dx as f32 * sx) as usize;
                let x1 = (((dx + 1) as f32 * sx) as usize).clamp(x0 + 1, w);
                let y0 = (dy as f32 * sy) as usize;
                let y1 = (((dy + 1) as f32 * sy) as usize).clamp(y0 + 1, h);
                let mut la = 0.0f32;
                let mut acc = [0.0f32; 3];
                let mut cnt = 0.0f32;
                for y in y0..y1 {
                    for x in x0..x1 {
                        la += luma[y * w + x];
                        let b = (y * w + x) * 3;
                        acc[0] += color[b] as f32 / 255.0;
                        acc[1] += color[b + 1] as f32 / 255.0;
                        acc[2] += color[b + 2] as f32 / 255.0;
                        cnt += 1.0;
                    }
                }
                *lo = la / cnt;
                rgb[0] = acc[0] / cnt;
                rgb[1] = acc[1] / cnt;
                rgb[2] = acc[2] / cnt;
            });

        // Bilateral blur on the working grid. The radius is authored at the
        // 1200 px reference (2 px); on a sub-reference proxy it shrinks
        // so the relative effect still matches the full render, and at or above
        // the reference it stays 2 (unchanged).
        let r = ((2.0 * Self::detail_radius_factor(w, h)).round() as isize).max(1);
        let k = (2 * r + 1) as usize;
        let mut weights = vec![0.0f32; k * k];
        for dy in -r..=r {
            for dx in -r..=r {
                let dist_sq = (dx * dx + dy * dy) as f32;
                weights[((dy + r) as usize) * k + (dx + r) as usize] = (-dist_sq / 2.0).exp();
            }
        }
        let mut out = vec![0.0f32; sw * sh * 3];
        out.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
            let cx = i % sw;
            let cy = i / sw;
            let l_c = small_luma[i];
            let mut acc = [0.0f32; 3];
            let mut wsum = 0.0f32;
            for dy in -r..=r {
                let ny = (cy as isize + dy).clamp(0, sh as isize - 1) as usize;
                for dx in -r..=r {
                    let nx = (cx as isize + dx).clamp(0, sw as isize - 1) as usize;
                    let ni = ny * sw + nx;
                    let diff = (l_c - small_luma[ni]).abs();
                    let range_w = 1.0 / (1.0 + diff * diff * 40.0);
                    let wgt = weights[((dy + r) as usize) * k + (dx + r) as usize] * range_w;
                    acc[0] += small_rgb[ni * 3] * wgt;
                    acc[1] += small_rgb[ni * 3 + 1] * wgt;
                    acc[2] += small_rgb[ni * 3 + 2] * wgt;
                    wsum += wgt;
                }
            }
            o[0] = acc[0] / wsum;
            o[1] = acc[1] / wsum;
            o[2] = acc[2] / wsum;
        });

        TextureBase {
            data: out,
            sw,
            sh,
        }
    }

    /// Reduced-resolution Gaussian RGB base, sampled bilinearly by the pixel
    /// loop. Used by Glow, Structure and Halation as a low-pass reference.
    ///
    /// The source is box-downsampled in **linear** light (so highlight energy
    /// is preserved) and stored back in display sRGB, then each channel is
    /// blurred with a resolution-independent radius.
    fn compute_blur_base(
        color: &[u8],
        w: usize,
        h: usize,
        work_max: usize,
        radius_frac: f32,
    ) -> TextureBase {
        let scale = (w.max(h) as f32 / work_max as f32).max(1.0);
        let sw = ((w as f32 / scale).round() as usize).max(1);
        let sh = ((h as f32 / scale).round() as usize).max(1);
        let sx = w as f32 / sw as f32;
        let sy = h as f32 / sh as f32;

        let mut small = vec![0.0f32; sw * sh * 3];
        small.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
            let dx = i % sw;
            let dy = i / sw;
            let x0 = (dx as f32 * sx) as usize;
            let x1 = (((dx + 1) as f32 * sx) as usize).clamp(x0 + 1, w);
            let y0 = (dy as f32 * sy) as usize;
            let y1 = (((dy + 1) as f32 * sy) as usize).clamp(y0 + 1, h);
            let mut acc = [0.0f32; 3];
            let mut cnt = 0.0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    let b = (y * w + x) * 3;
                    acc[0] += srgb_to_linear(color[b] as f32 / 255.0);
                    acc[1] += srgb_to_linear(color[b + 1] as f32 / 255.0);
                    acc[2] += srgb_to_linear(color[b + 2] as f32 / 255.0);
                    cnt += 1.0;
                }
            }
            let inv = 1.0 / cnt.max(1.0);
            o[0] = linear_to_srgb(acc[0] * inv);
            o[1] = linear_to_srgb(acc[1] * inv);
            o[2] = linear_to_srgb(acc[2] * inv);
        });

        let radius = (radius_frac * sw.min(sh) as f32).max(1.0);
        let mut data = vec![0.0f32; sw * sh * 3];
        for c in 0..3 {
            let plane: Vec<f32> = small.par_chunks(3).map(|p| p[c]).collect();
            let blurred = Self::gaussian_blur_map(&plane, sw, sh, radius);
            data.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
                o[c] = blurred[i];
            });
        }

        TextureBase {
            data,
            sw,
            sh,
        }
    }

    /// Reduced-resolution base for noise reduction: Oklab `a`/`b` planes (wide
    /// low-pass) plus a guided log2-luminance guide. Works at the same reduced
    /// resolution as the Texture base, so the radius stays scale-invariant.
    fn compute_noise_base(color: &[u8], luma: &[f32], w: usize, h: usize, scale: f32) -> NoiseBase {
        let sw = ((w as f32 / scale).round() as usize).max(1);
        let sh = ((h as f32 / scale).round() as usize).max(1);
        let sx = w as f32 / sw as f32;
        let sy = h as f32 / sh as f32;

        // Downsample linear RGB and luma. Oklab is stored per small pixel as
        // `[a, b, 0]` (the third slot is unused here).
        let mut small_lab = vec![0.0f32; sw * sh * 3];
        let mut small_luma = vec![0.0f32; sw * sh];
        small_luma
            .par_iter_mut()
            .zip(small_lab.par_chunks_mut(3))
            .enumerate()
            .for_each(|(i, (lo, lab))| {
                let dx = i % sw;
                let dy = i / sw;
                let x0 = (dx as f32 * sx) as usize;
                let x1 = (((dx + 1) as f32 * sx) as usize).clamp(x0 + 1, w);
                let y0 = (dy as f32 * sy) as usize;
                let y1 = (((dy + 1) as f32 * sy) as usize).clamp(y0 + 1, h);
                let (mut la, mut cnt) = (0.0f32, 0.0f32);
                let mut acc = [0.0f32; 3];
                for y in y0..y1 {
                    for x in x0..x1 {
                        la += luma[y * w + x];
                        let b = (y * w + x) * 3;
                        // Average in linear light, not sRGB.
                        acc[0] += srgb_to_linear(color[b] as f32 / 255.0);
                        acc[1] += srgb_to_linear(color[b + 1] as f32 / 255.0);
                        acc[2] += srgb_to_linear(color[b + 2] as f32 / 255.0);
                        cnt += 1.0;
                    }
                }
                *lo = la / cnt;
                let oklab = rgb_to_oklab(acc[0] / cnt, acc[1] / cnt, acc[2] / cnt);
                lab[0] = oklab[1];
                lab[1] = oklab[2];
            });

        // Radii authored at the 1200 px reference; shrunk on a sub-reference
        // proxy so the relative effect matches the full render.
        let radius_factor = Self::detail_radius_factor(w, h);
        let chroma_radius = (noise::CHROMA_RADIUS * radius_factor).max(1.0);
        let luma_radius = (noise::LUMA_RADIUS * radius_factor).max(1.0);

        // Wide chroma low-pass (a and b separately).
        let a_plane: Vec<f32> = small_lab.par_chunks(3).map(|p| p[0]).collect();
        let b_plane: Vec<f32> = small_lab.par_chunks(3).map(|p| p[1]).collect();
        let a_plane = Self::gaussian_blur_map(&a_plane, sw, sh, chroma_radius);
        let b_plane = Self::gaussian_blur_map(&b_plane, sw, sh, chroma_radius);

        // Edge-aware log2-luminance guide (self-guided filter).
        let guide = Self::guided_log_luma(&small_luma, sw, sh, luma_radius, noise::LUMA_EPS);

        let mut data = vec![0.0f32; sw * sh * 3];
        data.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
            o[0] = a_plane[i];
            o[1] = b_plane[i];
            o[2] = guide[i];
        });

        NoiseBase { data, sw, sh }
    }

    /// Self-guided filter (He et al.) on log2-luminance, one plane. Used as the
    /// edge-aware luminance guide for noise reduction.
    fn guided_log_luma(luma: &[f32], w: usize, h: usize, radius: f32, eps: f32) -> Vec<f32> {
        let n = w * h;
        let p: Vec<f32> = luma.par_iter().map(|y| y.max(1e-4).log2()).collect();
        let mean = Self::gaussian_blur_map(&p, w, h, radius);
        let p2: Vec<f32> = p.par_iter().map(|v| v * v).collect();
        let corr = Self::gaussian_blur_map(&p2, w, h, radius);
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        a.par_iter_mut()
            .zip(b.par_iter_mut())
            .enumerate()
            .for_each(|(i, (ai, bi))| {
                let var = (corr[i] - mean[i] * mean[i]).max(0.0);
                let g = var / (var + eps);
                *ai = g;
                *bi = mean[i] - g * mean[i];
            });
        let a_mean = Self::gaussian_blur_map(&a, w, h, radius);
        let b_mean = Self::gaussian_blur_map(&b, w, h, radius);
        (0..n).map(|i| a_mean[i] * p[i] + b_mean[i]).collect()
    }

    /// Bilinear sampling of the noise base (`[a, b, log2 guide]`).
    fn sample_noise_base(base: &NoiseBase, x: usize, y: usize, w: usize, h: usize) -> [f32; 3] {
        let (sw, sh) = (base.sw, base.sh);
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
            let v00 = base.data[(ya * sw + xa) * 3 + c];
            let v10 = base.data[(ya * sw + xb) * 3 + c];
            let v01 = base.data[(yb * sw + xa) * 3 + c];
            let v11 = base.data[(yb * sw + xb) * 3 + c];
            let top = v00 * (1.0 - tx) + v10 * tx;
            let bot = v01 * (1.0 - tx) + v11 * tx;
            out[c] = top * (1.0 - ty) + bot * ty;
        }
        out
    }

    /// Shared detail-mask base: a reduced-resolution `0..=1` gate, ~0 on flat /
    /// noisy areas and ~1 on texture/edges. Computed from a local high-pass
    /// energy so fine noise averages out and the gate follows real structure.
    fn compute_mask_base(luma: &[f32], w: usize, h: usize, scale: f32) -> MaskBase {
        let sw = ((w as f32 / scale).round() as usize).max(1);
        let sh = ((h as f32 / scale).round() as usize).max(1);
        let sx = w as f32 / sw as f32;
        let sy = h as f32 / sh as f32;

        // Box-downsample luma.
        let mut small = vec![0.0f32; sw * sh];
        small.par_iter_mut().enumerate().for_each(|(i, out)| {
            let dx = i % sw;
            let dy = i / sw;
            let x0 = (dx as f32 * sx) as usize;
            let x1 = (((dx + 1) as f32 * sx) as usize).clamp(x0 + 1, w);
            let y0 = (dy as f32 * sy) as usize;
            let y1 = (((dy + 1) as f32 * sy) as usize).clamp(y0 + 1, h);
            let mut acc = 0.0f32;
            let mut cnt = 0.0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    acc += luma[y * w + x];
                    cnt += 1.0;
                }
            }
            *out = acc / cnt;
        });

        // Local high-pass energy, then a soft gate. The radius is authored at
        // the 1200 px reference; shrunk on a sub-reference proxy.
        let mask_radius = (MASK_RADIUS * Self::detail_radius_factor(w, h)).max(1.0);
        let blur = Self::gaussian_blur_map(&small, sw, sh, mask_radius);
        let hp2: Vec<f32> = small
            .par_iter()
            .zip(blur.par_iter())
            .map(|(s, b)| {
                let hp = s - b;
                hp * hp
            })
            .collect();
        let energy = Self::gaussian_blur_map(&hp2, sw, sh, mask_radius);
        let data: Vec<f32> = energy.par_iter().map(|en| en / (en + MASK_KNEE2)).collect();

        MaskBase { data, sw, sh }
    }

    /// Bilinear sampling of the shared detail mask.
    fn sample_mask_base(base: &MaskBase, x: usize, y: usize, w: usize, h: usize) -> f32 {
        let (sw, sh) = (base.sw, base.sh);
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
        let v00 = base.data[ya * sw + xa];
        let v10 = base.data[ya * sw + xb];
        let v01 = base.data[yb * sw + xa];
        let v11 = base.data[yb * sw + xb];
        let top = v00 * (1.0 - tx) + v10 * tx;
        let bot = v01 * (1.0 - tx) + v11 * tx;
        top * (1.0 - ty) + bot * ty
    }

    /// Luma map from an RGB buffer.
    pub(crate) fn compute_luma_map(buffer: &[u8]) -> Vec<f32> {
        buffer
            .par_chunks(3)
            .map(|p| {
                0.2126 * (p[0] as f32 / 255.0)
                    + 0.7152 * (p[1] as f32 / 255.0)
                    + 0.0722 * (p[2] as f32 / 255.0)
            })
            .collect()
    }

    /// Downsample + bilateral blur of lightness for Clarity.
    /// Returns (map, width, height).
    pub(crate) fn compute_clarity_cache(
        luma_map: &[f32],
        w: usize,
        h: usize,
    ) -> (Vec<f32>, usize, usize) {
        // The downsample factor is normalized to resolution (see `clarity_scale`)
        // so the Clarity base has the same frequency in preview and export.
        let scale = Self::clarity_scale(w, h);
        let sw = w / scale + 1;
        let sh = h / scale + 1;

        // 1. Downsample luma_map → small_luma
        let mut small_luma = vec![0.0f32; sw * sh];
        for y in 0..sh {
            for x in 0..sw {
                let mut sum = 0.0f32;
                let mut count = 0.0f32;
                for dy in 0..scale {
                    for dx in 0..scale {
                        let px = (x * scale + dx).min(w - 1);
                        let py = (y * scale + dy).min(h - 1);
                        sum += luma_map[py * w + px];
                        count += 1.0;
                    }
                }
                small_luma[y * sw + x] = sum / count;
            }
        }

        // 2. Bilateral blur small_luma → clarity_cache. Radius authored at the
        // 1200 px reference, shrunk on a sub-reference proxy.
        let mut clarity_cache = vec![0.0f32; sw * sh];
        let r = ((2.0 * Self::detail_radius_factor(w, h)).round() as isize).max(1);
        clarity_cache
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, out)| {
                let cx = (idx % sw) as isize;
                let cy = (idx / sw) as isize;
                let center_luma = small_luma
                    [cy.min(sh as isize - 1) as usize * sw + cx.min(sw as isize - 1) as usize];

                let mut sum = 0.0f32;
                let mut count = 0.0f32;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let nx = (cx + dx).clamp(0, sw as isize - 1) as usize;
                        let ny = (cy + dy).clamp(0, sh as isize - 1) as usize;
                        let neighbor_luma = small_luma[ny * sw + nx];

                        let spatial_w = 1.0 / (1.0 + (dx * dx + dy * dy) as f32 * 0.5);
                        let luma_diff = (center_luma - neighbor_luma).abs();
                        let range_w = 1.0 / (1.0 + luma_diff * luma_diff * 250.0);
                        let w = spatial_w * range_w;
                        sum += neighbor_luma * w;
                        count += w;
                    }
                }
                *out = sum / count;
            });
        (clarity_cache, sw, sh)
    }

    /// Full offline render of an arbitrary image: the retouch layer, the color
    /// pass (sliders + external LUT) and the spatial pass (Texture, Clarity,
    /// Sharpening, Grain). Used by export so that a single save and batch
    /// processing produce the SAME result at full resolution.
    pub(crate) fn render_full_image(
        img: &RgbImage,
        settings: &FilterSettings,
        lut_path: &Option<PathBuf>,
        layer: &RetouchLayer,
    ) -> Vec<u8> {
        // ---- Retouch (heal) runs BEFORE everything else, on the raw base ----
        let layered;
        let src = if layer.is_empty() {
            img
        } else {
            let mut healed = img.clone();
            layer.apply_to(&mut healed);
            layered = healed;
            &layered
        };
        // Export always bakes the 64³ LUT (`bake_temp_lut` returns `Some`).
        let lut = Self::bake_temp_lut(settings, lut_path);
        Self::render_base_through_lut(src, settings, lut.as_ref())
    }

    /// Pure preview render of one already-healed base at its working
    /// resolution. Reproduces the interactive preview's cold, non-drag path — the
    /// color pass through the always-[`PREVIEW_LUT_SIZE`]³ preview LUT (with the
    /// same identity fast path as [`Self::process_preview`]), full-quality dehaze,
    /// then the spatial chain — so the background proxy worker and the UI produce
    /// byte-identical pixels. Never uses the export 64³ LUT.
    pub(crate) fn render_preview_at(
        base: &RgbImage,
        settings: &FilterSettings,
        active_lut: Option<&Lut3D>,
    ) -> Vec<u8> {
        // Default sliders + no external LUT = an identity pass; the UI skips the
        // bake entirely there, so the proxy render must too (parity).
        let lut = if *settings == FilterSettings::default() && active_lut.is_none() {
            None
        } else {
            Some(Self::bake_lut(settings, active_lut, PREVIEW_LUT_SIZE))
        };
        Self::render_base_through_lut(base, settings, lut.as_ref())
    }

    /// Passes 1-3 shared by [`Self::render_full_image`] and
    /// [`Self::render_preview_at`]: color through the supplied LUT (or an identity
    /// copy when `lut` is `None`), dehaze, then the spatial chain. One
    /// implementation is what keeps the export path, the UI preview and the
    /// background proxy render from diverging.
    pub(crate) fn render_base_through_lut(
        img: &RgbImage,
        settings: &FilterSettings,
        lut: Option<&Lut3D>,
    ) -> Vec<u8> {
        let w = img.width() as usize;
        let h = img.height() as usize;
        let mut color_buffer = img.as_raw().clone();

        // ---- Pass 1: color (through the supplied LUT) ----
        if let Some(lut) = lut {
            color_buffer.par_chunks_mut(3).for_each(|pixel| {
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;
                let mapped = lut.apply(r, g, b);
                pixel[0] = (mapped[0] * 255.0).clamp(0.0, 255.0) as u8;
                pixel[1] = (mapped[1] * 255.0).clamp(0.0, 255.0) as u8;
                pixel[2] = (mapped[2] * 255.0).clamp(0.0, 255.0) as u8;
            });
        }

        // ---- Pass 2: dehaze (spatial, Dark Channel Prior) ----
        if settings.dehaze != 0.0 {
            let mut dh = vec![0u8; color_buffer.len()];
            crate::pipeline::dehaze::apply(
                &color_buffer,
                &mut dh,
                w,
                h,
                settings.dehaze / 100.0,
            );
            color_buffer = dh;
        }

        let needs_spatial = settings.texture != 0.0
            || settings.clarity != 0.0
            || settings.structure != 0.0
            || settings.sharpen > 0.0
            || settings.glow > 0.0
            || settings.halation > 0.0
            || settings.flare > 0.0
            || settings.grain != 0.0
            || !settings.noise.is_identity()
            || !settings.vignette.is_identity();

        if !needs_spatial {
            return color_buffer;
        }

        // ---- Pass 3: spatial effects ----
        let luma_map = Self::compute_luma_map(&color_buffer);
        let (clarity_cache, csw, csh) = if settings.clarity != 0.0 {
            Self::compute_clarity_cache(&luma_map, w, h)
        } else {
            (Vec::new(), 0, 0)
        };
        let grain_map = Self::generate_grain_map(
            w,
            h,
            42,
            settings.grain_size,
            settings.grain_roughness > 0.0,
        );

        let mut output = vec![0u8; color_buffer.len()];
        Self::run_spatial_pass(
            settings,
            &color_buffer,
            &mut output,
            &grain_map,
            &luma_map,
            &clarity_cache,
            csw,
            csh,
            w,
            h,
        );
        output
    }

    pub(crate) fn process_preview(&mut self, ctx: &egui::Context) {
        // Rebuild the working proxy first (debounced); it may mark the
        // pipeline dirty at the new resolution.
        self.maybe_rebuild_proxy(ctx);
        let needs_reprocess = self.color_dirty || self.spatial_dirty || self.retouch_dirty;
        if !needs_reprocess && !self.full_render_pending {
            return;
        }

        // ==========================================
        // DEBOUNCE: during drag do not render more often than 60fps
        // If full_render_pending — do not debounce (a render is needed after release)
        // ==========================================
        if !self.full_render_pending {
            let now = Instant::now();
            if self.drag_active
                && (now - self.last_render_time) < std::time::Duration::from_millis(16)
            {
                ctx.request_repaint();
                return;
            }
            self.last_render_time = now;
        }
        self.full_render_pending = false;

        if self.preview_base.is_none() {
            return;
        }

        // Heal the base BEFORE Pass 1. Both the retouch mode and the normal
        // pipeline operate on this healed base.
        let (w, h) = {
            let base = self.preview_base.as_ref().unwrap();
            (base.width() as usize, base.height() as usize)
        };
        self.retouch_dirty = false;

        // Retouch mode: work and display at full resolution (long side ≤ 4096)
        // so real pixel detail is visible. The layer is still non-destructive;
        // the full-resolution buffers live only while the tool is active.
        if self.retouch.active {
            // Decode + whole-layer rebuild run on the background heal
            // worker; here we only enqueue (if needed) and upload whatever is
            // ready, so this branch returns immediately.
            let path = self.image_path.clone();
            self.request_full_base(ctx, path.as_deref());
            self.ensure_full_rebuild(ctx);
            self.upload_full_retouch_textures(ctx);
            self.color_dirty = false;
            self.spatial_dirty = false;
            self.color_buffer_valid = false;
            return;
        }

        let base_raw = {
            let base = self.preview_base.as_ref().unwrap();
            self.retouch.healed_base(base).as_raw().to_vec()
        };

        let needs_spatial = self.settings.texture != 0.0
            || self.settings.clarity != 0.0
            || self.settings.structure != 0.0
            || self.settings.sharpen > 0.0
            || self.settings.glow > 0.0
            || self.settings.halation > 0.0
            || self.settings.flare > 0.0
            || self.settings.grain > 0.0
            || !self.settings.noise.is_identity()
            || !self.settings.vignette.is_identity();

        // During a COLOR slider drag the spatial pass is not recomputed:
        // this saves ~22 ms/frame, the full render happens on release.
        let skip_spatial_for_drag = self.drag_active && self.color_dirty;
        let needs_spatial = needs_spatial && !skip_spatial_for_drag;

        // ---- Pass 1: Color (through the baked LUT) ----
        // `!color_buffer_valid` — the ready render was taken from the cache, but the color_buffer
        // for subsequent spatial edits is not yet assembled.
        if self.color_dirty || !self.color_buffer_valid {
            // The working resolution can change at runtime when the viewport proxy is
            // swapped, so `color_buffer` may still be sized for the previous
            // resolution: re-fit it before the copy. `resize` is a no-op when the
            // length already matches, so the common path is unchanged.
            if self.color_buffer.len() != base_raw.len() {
                self.color_buffer.resize(base_raw.len(), 0);
            }
            debug_assert_eq!(
                self.color_buffer.len(),
                base_raw.len(),
                "colour buffer must match the working resolution before the copy"
            );
            // Fast path: default sliders + no LUT = an identity pass.
            // We do not bake an identity LUT — that is a noticeable saving on every frame.
            if self.settings == FilterSettings::default() && self.active_lut.is_none() {
                self.color_buffer.copy_from_slice(&base_raw);
                self.combined_lut = None;
            } else {
                // The preview is always baked at `PREVIEW_LUT_SIZE` (33³): versus
                // 64³ the mean error is ~1/255 for both, and 64³ only wins in the
                // worst case of clipping — while costing ~18 ms on release. The
                // export stays at 64³ (`bake_temp_lut`), so files do not lose quality.
                self.combined_lut = self.bake_combined_lut(PREVIEW_LUT_SIZE);

                // Apply the LUT to all pixels (trilinear interpolation)
                self.color_buffer.copy_from_slice(&base_raw);
                if let Some(lut) = &self.combined_lut {
                    self.color_buffer.par_chunks_mut(3).for_each(|pixel| {
                        let r = pixel[0] as f32 / 255.0;
                        let g = pixel[1] as f32 / 255.0;
                        let b = pixel[2] as f32 / 255.0;
                        let mapped = lut.apply(r, g, b);
                        pixel[0] = (mapped[0] * 255.0).clamp(0.0, 255.0) as u8;
                        pixel[1] = (mapped[1] * 255.0).clamp(0.0, 255.0) as u8;
                        pixel[2] = (mapped[2] * 255.0).clamp(0.0, 255.0) as u8;
                    });
                } else {
                    // fallback (rare)
                    Self::run_color_pass(
                        &self.settings,
                        self.active_lut.as_deref(),
                        &mut self.color_buffer,
                    );
                }
            }
            self.color_buffer_valid = true;
            self.color_dirty = false;
            self.spatial_dirty = true;
            self.luma_cache_valid = false;
            self.clarity_cache_valid = false;
            self.dehaze_cache_valid = false;
            self.spatial_base_valid = false;
            // The spatial input changed — invalidate the drag tier.
            self.spatial_input_revision = self.spatial_input_revision.wrapping_add(1);
        }

        let has_dehaze = self.settings.dehaze != 0.0;
        let has_dehaze_active = has_dehaze && !skip_spatial_for_drag;
        // During drag the haze map is computed in fast mode, on release in
        // the full one. A mode change also requires a recompute (otherwise on release
        // the fast result would remain).
        let dehaze_fast = self.drag_active;
        let dehaze_changed = self.dehaze_applied != self.settings.dehaze
            || !self.dehaze_cache_valid
            || self.dehaze_cached_fast != dehaze_fast;

        // ---- Dehaze (spatial, Dark Channel Prior) ----
        if has_dehaze_active && dehaze_changed {
            let n = self.color_buffer.len();
            if self.dehazed_buffer.len() != n {
                self.dehazed_buffer.resize(n, 0);
            }
            if dehaze_fast {
                crate::pipeline::dehaze::apply_fast(
                    &self.color_buffer,
                    &mut self.dehazed_buffer,
                    w,
                    h,
                    self.settings.dehaze / 100.0,
                );
            } else {
                crate::pipeline::dehaze::apply(
                    &self.color_buffer,
                    &mut self.dehazed_buffer,
                    w,
                    h,
                    self.settings.dehaze / 100.0,
                );
            }
            self.dehaze_cache_valid = true;
            self.dehaze_applied = self.settings.dehaze;
            self.dehaze_cached_fast = dehaze_fast;
            // The spatial input changed — invalidate the drag tier.
            self.spatial_input_revision = self.spatial_input_revision.wrapping_add(1);
            // The spatial input changed, but during drag the Details precomputations
            // (luma/clarity/texture/sharpen) are NOT recomputed — they catch up
            // on release (where `color_dirty` already resets the caches). The spatial
            // pass itself still runs: the effects are visible, their base just
            // lags during the drag.
            if !dehaze_fast {
                self.luma_cache_valid = false;
                self.clarity_cache_valid = false;
                self.spatial_base_valid = false;
            }
        } else if !has_dehaze_active && self.dehaze_applied != self.settings.dehaze {
            // Dehaze is off (or skipped during drag): the spatial input = color_buffer.
            self.dehaze_applied = self.settings.dehaze;
            self.luma_cache_valid = false;
            self.clarity_cache_valid = false;
            self.spatial_base_valid = false;
            // The spatial input changed — invalidate the drag tier.
            self.spatial_input_revision = self.spatial_input_revision.wrapping_add(1);
        }

        // ---- Recompute luma_cache (only when the spatial input changed) ----
        if self.spatial_dirty && needs_spatial && !self.luma_cache_valid {
            let input: &[u8] = if has_dehaze_active {
                &self.dehazed_buffer
            } else {
                &self.color_buffer
            };
            self.luma_cache = input
                .par_chunks(3)
                .map(|p| {
                    0.2126 * (p[0] as f32 / 255.0)
                        + 0.7152 * (p[1] as f32 / 255.0)
                        + 0.0722 * (p[2] as f32 / 255.0)
                })
                .collect();
            self.luma_cache_valid = true;
        }

        // ---- Recompute clarity_cache (depends on luma_cache, also only on color change) ----
        if self.spatial_dirty
            && needs_spatial
            && self.settings.clarity != 0.0
            && !self.clarity_cache_valid
            && self.luma_cache_valid
        {
            let (cache, sw, sh) = Self::compute_clarity_cache(&self.luma_cache, w, h);
            self.clarity_cache = cache;
            self.clarity_cache_sw = sw;
            self.clarity_cache_sh = sh;
            self.clarity_cache_valid = true;
        }

        // ---- Recompute spatial_base (Texture/Sharpen) — also only on input change.
        // If the effect was off when the base was computed and is now on —
        // compute it: `None` in the cache with a non-zero strength means "needs computing".
        if self.spatial_dirty
            && needs_spatial
            && self.luma_cache_valid
            && (!self.spatial_base_valid
                || (self.settings.texture != 0.0 && self.spatial_base.texture_base.is_none())
                || (self.settings.sharpen > 0.0 && self.spatial_base.sharpen_blur.is_none())
                || (!self.settings.noise.is_identity() && self.spatial_base.noise_base.is_none())
                || (self.settings.detail_masking > 0.0
                    && self.spatial_base.mask_base.is_none())
                || ((self.settings.glow > 0.0 || self.settings.structure != 0.0)
                    && self.spatial_base.wide_blur.is_none())
                || (self.settings.halation > 0.0
                    && self.spatial_base.halation_blur.is_none())
                || (self.settings.flare > 0.0 && self.spatial_base.flare_base.is_none()))
        {
            let input: &[u8] = if has_dehaze_active {
                &self.dehazed_buffer
            } else {
                &self.color_buffer
            };
            self.spatial_base =
                Self::compute_spatial_base(&self.settings, input, &self.luma_cache, w, h);
            self.spatial_base_valid = true;
        }

        // Keep the grain map in sync with the Grain Size slider before the
        // spatial pass below borrows it. Skipped while the reduced-resolution
        // drag tier is active — it maintains its own grain map at the drag
        // resolution, so regenerating the full-resolution one every drag frame
        // (the dominant cost of a Grain Size drag) would be wasted.
        let drag_tier_active = self.spatial_dirty && needs_spatial && self.drag_active;
        if self.settings.grain != 0.0 && !drag_tier_active {
            self.ensure_grain_map(w, h);
        }

        // ---- Pass: Spatial effects ----
        //
        // While a slider is being dragged and the spatial pass actually
        // needs to run, use the reduced-resolution drag tier (the staged pipeline
        // on a downscaled input) instead of the full-resolution pass. The
        // authoritative full-resolution render runs on release via
        // `full_render_pending`.
        let mut drag_uploaded = false;
        if self.spatial_dirty {
            if needs_spatial && self.drag_active {
                self.render_spatial_drag_tier(ctx, w, h, has_dehaze_active);
                drag_uploaded = true;
            } else {
                let input: &[u8] = if has_dehaze_active {
                    &self.dehazed_buffer
                } else {
                    &self.color_buffer
                };
                if needs_spatial {
                    Self::run_spatial_pass_with_base(
                        &self.settings,
                        input,
                        &mut self.processed_pixels,
                        &self.grain_map,
                        &self.clarity_cache,
                        self.clarity_cache_sw,
                        self.clarity_cache_sh,
                        &self.spatial_base,
                        w,
                        h,
                    );
                } else {
                    if self.processed_pixels.len() != input.len() {
                        self.processed_pixels.resize(input.len(), 0);
                    }
                    self.processed_pixels.copy_from_slice(input);
                }
            }
            self.spatial_dirty = false;
        }

        if !drag_uploaded {
            // GPU upload (full-resolution preview)
            self.processed_revision = self.processed_revision.wrapping_add(1);
            self.upload_preview_textures(ctx);
        }

        // Store the ready render in the cache (final only, not during drag).
        // The identity pass (default + no LUT) is not cached: the result equals base,
        // and a separate Vec would only take up budget. On return such a
        // frame is recomputed with two copies (cheap).
        let is_identity = self.settings == FilterSettings::default() && self.lut_path.is_none();
        if !self.drag_active && !is_identity {
            if let Some(path) = self.image_path.clone() {
                self.preview_cache.store_render(
                    &path,
                    self.settings,
                    self.lut_path.clone(),
                    (w, h),
                    self.processed_pixels.clone(),
                );
            }
        }

    }

    /// Uploads `processed_pixels` and the base frame into GPU textures, applies
    /// the clipping warning overlay, and refreshes the output histogram.
    ///
    /// Upload only what changed — the display texture is re-uploaded only
    /// when the processed pixels or the warning overlay changed, the original
    /// (before) texture only when the source image changes, and the histogram only
    /// when the rendered pixels change. The clipping overlay paints into a reused
    /// scratch buffer instead of cloning.
    pub(crate) fn upload_preview_textures(&mut self, ctx: &egui::Context) {
        let (w, h) = match self.preview_base.as_ref() {
            Some(b) => (b.width() as usize, b.height() as usize),
            None => return,
        };

        // The texture stays raw: orientation and straighten are drawn on the GPU
        // (a rotated quad mesh), so changing the angle is free and smooth.
        let has_render = self.processed_pixels.len() == w * h * 3;

        // Clipping warnings replace the shown pixels with the warning colours.
        // They are a view aid, not an edit, so `processed_pixels` is untouched.
        let overlay = self.clipping.overlay();

        // --- Display texture: only when the shown pixels or the overlay changed.
        let display_key = (self.processed_revision, w as u32, h as u32, overlay);
        let display_changed =
            self.texture.is_none() || self.upload_display_key != Some(display_key);
        if display_changed {
            let proc: &[u8] = if overlay.any() {
                // Reuse the scratch buffer (no per-frame clone/to_vec).
                let src: &[u8] = if has_render {
                    &self.processed_pixels
                } else {
                    self.preview_base.as_ref().unwrap().as_raw()
                };
                let mut buf = std::mem::take(&mut self.scratch_clip);
                buf.clear();
                buf.extend_from_slice(src);
                overlay.paint(&mut buf);
                self.scratch_clip = buf;
                &self.scratch_clip
            } else if has_render {
                &self.processed_pixels
            } else {
                self.preview_base.as_ref().unwrap().as_raw()
            };
            let color_image = egui::ColorImage::from_rgb([w, h], proc);
            if let Some(tex) = &mut self.texture {
                tex.set(color_image, egui::TextureOptions::LINEAR);
            } else {
                self.texture =
                    Some(ctx.load_texture("preview", color_image, egui::TextureOptions::LINEAR));
            }
            self.upload_display_key = Some(display_key);
        }

        // --- Output histogram: only when the rendered pixels changed (not on an
        // overlay-only re-upload).
        if self.histogram_revision != self.processed_revision {
            self.histogram = if has_render {
                super::histogram::Histogram::of_rgb8(&self.processed_pixels)
            } else {
                super::histogram::Histogram::of_rgb8(self.preview_base.as_ref().unwrap().as_raw())
            };
            self.histogram_revision = self.processed_revision;
        }
        self.clip_overlay_applied = overlay;

        // --- Original (before) texture: the base frame is unchanged across most
        // preview frames; upload only when the source image changes.
        let orig_key = self.image_path.clone().map(|p| (p, w as u32, h as u32));
        let orig_changed = self.original_texture.is_none()
            || match (&orig_key, &self.upload_orig_key) {
                (Some(k), Some(prev)) => k != prev,
                _ => true,
            };
        if orig_changed {
            let orig_image = {
                let base = self.preview_base.as_ref().unwrap();
                egui::ColorImage::from_rgb([w, h], base.as_raw())
            };
            if let Some(tex) = &mut self.original_texture {
                tex.set(orig_image, egui::TextureOptions::LINEAR);
            } else {
                self.original_texture =
                    Some(ctx.load_texture("original", orig_image, egui::TextureOptions::LINEAR));
            }
            self.upload_orig_key = orig_key;
        }
    }
}

// ==========================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic RGB frame, distinct per pixel, so a length or content
    /// mismatch between resolutions cannot pass silently.
    fn synth_rgb(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            let v = ((x * 7 + y * 3) % 256) as u8;
            image::Rgb([v, 255 - v, v / 2 + 30])
        })
    }

    /// A mid-session working-resolution change must not desync the cached colour
    /// buffer. The viewport proxy swaps `preview_base` to a new resolution while
    /// `color_buffer` still holds the old one; the colour pass then copies the new
    /// base into it. Before the fix that copy panicked (`copy_from_slice` length
    /// mismatch — `panic = "abort"` in release, so an instant process kill). This
    /// drives the real stateful `process_preview` across two resolutions, the shape
    /// the `Harness` in `perf.rs` cannot reach, and pins the stateful result to the
    /// pure `render_preview_at` output at both sizes.
    #[test]
    fn resolution_swap_refits_the_colour_buffer() {
        let ctx = egui::Context::default();
        let mut app = TinyLumaApp::new_headless();
        // Colour-only settings: the colour-copy guard is the code under test, and a
        // non-default LUT keeps the identity fast path out of the way.
        app.settings.exposure = 20.0;
        app.settings.contrast = 15.0;
        app.settings.temp = 12.0;
        assert_ne!(app.settings, FilterSettings::default());

        let base_a = synth_rgb(64, 48);

        // Mirror the load path in image_io: both resolution-keyed buffers start
        // sized to the loaded frame, so the first render is clean and any failure
        // below is the swap itself, not the initial sizing.
        app.color_buffer = base_a.as_raw().to_vec();
        app.processed_pixels = base_a.as_raw().to_vec();

        // First render at resolution A.
        app.preview_base = Some(base_a.clone());
        app.proxy_dims = (64, 48);
        app.desired_proxy = (64, 48);
        app.color_dirty = true;
        app.spatial_dirty = true;
        app.process_preview(&ctx);

        let expect_a =
            TinyLumaApp::render_preview_at(&base_a, &app.settings, app.active_lut.as_deref());
        assert_eq!(
            app.color_buffer.len(),
            64 * 48 * 3,
            "colour buffer sized to A"
        );
        assert_eq!(
            app.processed_pixels, expect_a,
            "stateful render != pure render at A"
        );

        // Swap the working proxy to a different resolution, exactly as
        // `apply_proxy_reply` does when a worker reply lands.
        let base_b = synth_rgb(37, 29);
        app.preview_base = Some(base_b.clone());
        app.proxy_dims = (37, 29);
        app.desired_proxy = (37, 29);
        app.invalidate_resolution_caches(base_b.as_raw().len());
        app.color_dirty = true;
        app.spatial_dirty = true;

        // Must not panic on the colour-buffer copy, and must re-fit both buffers.
        app.process_preview(&ctx);

        let expect_b =
            TinyLumaApp::render_preview_at(&base_b, &app.settings, app.active_lut.as_deref());
        assert_eq!(
            app.color_buffer.len(),
            base_b.as_raw().len(),
            "colour buffer re-fitted to B"
        );
        assert_eq!(
            app.processed_pixels, expect_b,
            "stateful render != pure render at B"
        );
    }

    #[test]
    fn default_settings_roundtrip() {
        let mut buf: Vec<u8> = (0u8..=255).flat_map(|v| [v, 255 - v, v / 2]).collect();
        let orig = buf.clone();
        TinyLumaApp::run_color_pass(&FilterSettings::default(), None, &mut buf);
        for (a, b) in buf.iter().zip(orig.iter()) {
            assert!((*a as i32 - *b as i32).abs() <= 1, "changed {a} vs {b}");
        }
    }

    #[test]
    fn tone_curve_halves_the_output() {
        // A master curve ending at 0.5 maps white to mid-grey.
        let mut buf = vec![255u8, 255, 255];
        let mut s = FilterSettings::default();
        s.curves.master = crate::pipeline::curve::Curve::from_points(&[[0.0, 0.0], [1.0, 0.5]]);
        TinyLumaApp::run_color_pass(&s, None, &mut buf);
        assert!((buf[0] as i32 - 128).abs() <= 3, "not halved: {}", buf[0]);
    }

    #[test]
    fn tone_curve_bakes_into_the_lut() {
        let mut s = FilterSettings::default();
        s.curves.master = crate::pipeline::curve::Curve::from_points(&[[0.0, 0.0], [1.0, 0.5]]);
        let lut = TinyLumaApp::bake_lut(&s, None, 33);
        let mapped = lut.apply(1.0, 1.0, 1.0);
        assert!((mapped[0] - 0.5).abs() < 0.02, "{mapped:?}");
    }

    #[test]
    fn hsl_mixer_desaturates_red_only() {
        let mut red = vec![255u8, 0, 0];
        let mut blue = vec![0u8, 0, 255];
        let mut s = FilterSettings::default();
        s.hsl.bands[0][1] = -100.0; // desaturate the red band
        TinyLumaApp::run_color_pass(&s, None, &mut red);
        TinyLumaApp::run_color_pass(&s, None, &mut blue);
        assert!(
            (red[0] as i32 - red[1] as i32) < 20 && (red[0] as i32 - red[2] as i32) < 20,
            "red not desaturated: {red:?}"
        );
        assert!(
            blue[0] < 3 && blue[1] < 3 && blue[2] > 252,
            "blue must be untouched: {blue:?}"
        );
    }

    #[test]
    fn hsl_mixer_bakes_into_the_lut() {
        let mut s = FilterSettings::default();
        s.hsl.bands[0][1] = -100.0;
        let lut = TinyLumaApp::bake_lut(&s, None, 33);
        let mapped = lut.apply(1.0, 0.0, 0.0);
        assert!(mapped[0] - mapped[1] < 0.1, "red band not in LUT: {mapped:?}");
    }

    #[test]
    fn exposure_brightens() {
        let mut buf = vec![64u8, 64, 64];
        let mut s = FilterSettings::default();
        s.exposure = 1.0;
        TinyLumaApp::run_color_pass(&s, None, &mut buf);
        assert!(buf[0] > 64, "not brightened: {}", buf[0]);
    }

    #[test]
    fn warm_shifts_neutral_to_red() {
        let mut buf = vec![128u8, 128, 128];
        let mut s = FilterSettings::default();
        s.temp = 80.0;
        TinyLumaApp::run_color_pass(&s, None, &mut buf);
        assert!(buf[0] > buf[2], "not warm: {buf:?}");
    }

    #[test]
    fn saturation_minus_one_is_grey() {
        let mut buf = vec![200u8, 60, 20];
        let mut s = FilterSettings::default();
        s.saturation = -100.0;
        TinyLumaApp::run_color_pass(&s, None, &mut buf);
        let (r, g, b) = (buf[0] as i32, buf[1] as i32, buf[2] as i32);
        assert!((r - g).abs() <= 1 && (g - b).abs() <= 1, "not grey: {buf:?}");
    }

    #[test]
    fn highlights_plus_brightens_brights() {
        let mut buf = vec![230u8, 230, 230];
        let mut s = FilterSettings::default();
        s.highlights = 60.0;
        TinyLumaApp::run_color_pass(&s, None, &mut buf);
        assert!(buf[0] > 230, "highlights+ should brighten: {}", buf[0]);
    }

    #[test]
    fn highlights_minus_recovers_brights() {
        let mut buf = vec![230u8, 230, 230];
        let mut s = FilterSettings::default();
        s.highlights = -60.0;
        TinyLumaApp::run_color_pass(&s, None, &mut buf);
        assert!(buf[0] < 230, "highlights- should recover: {}", buf[0]);
    }

    #[test]
    fn creative_effects_change_output() {
        let (w, h) = (64usize, 48usize);
        let mut color = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let b = (y * w + x) * 3;
                let v = (x as f32 / w as f32 * 255.0) as u8;
                color[b] = v;
                color[b + 1] = v;
                color[b + 2] = v;
            }
        }
        // A blown highlight that the glow/halation/flare references can find.
        for y in 20..28 {
            for x in 28..36 {
                let b = (y * w + x) * 3;
                color[b] = 255;
                color[b + 1] = 255;
                color[b + 2] = 255;
            }
        }

        let luma = TinyLumaApp::compute_luma_map(&color);
        let (clarity, sw, sh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
        let grain = vec![0.0f32; w * h * 2];

        let mut s = FilterSettings::default();
        s.structure = 60.0;
        s.glow = 80.0;
        s.halation = 70.0;
        s.flare = 80.0;

        let mut out = vec![0u8; w * h * 3];
        TinyLumaApp::run_spatial_pass(
            &s, &color, &mut out, &grain, &luma, &clarity, sw, sh, w, h,
        );

        let diff = out
            .iter()
            .zip(color.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(diff > 0, "creative effects did not change the frame");
    }

    #[test]
    fn detail_scale_and_radius_factor_track_reference() {
        // Downsample factor stays >= 1: it never upsamples the working grid.
        assert!((TinyLumaApp::detail_scale(1200, 800) - 1.0).abs() < 1e-6);
        assert!((TinyLumaApp::detail_scale(800, 600) - 1.0).abs() < 1e-6);
        assert!((TinyLumaApp::detail_scale(6000, 4000) - 5.0).abs() < 1e-6);
        // Radius factor caps at 1.0 (reference and above) and shrinks below the
        // reference so a sub-1200 proxy keeps the same relative effect.
        assert!((TinyLumaApp::detail_radius_factor(1200, 800) - 1.0).abs() < 1e-6);
        assert!((TinyLumaApp::detail_radius_factor(6000, 4000) - 1.0).abs() < 1e-6);
        assert!((TinyLumaApp::detail_radius_factor(600, 400) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn clarity_scale_tracks_resolution() {
        assert_eq!(TinyLumaApp::clarity_scale(1200, 800), 8);
        assert_eq!(TinyLumaApp::clarity_scale(6000, 4000), 40);
    }

    #[test]
    fn gaussian_blur_radius1_is_binomial() {
        // With radius = 1 the weights must match the 1-2-1 kernel.
        let (w, h) = (5usize, 1usize);
        let mut src = vec![0.0f32; w * h];
        src[2] = 1.0;
        let out = TinyLumaApp::gaussian_blur_map(&src, w, h, 1.0);
        assert!((out[1] - 0.25).abs() < 5e-3, "left {}", out[1]);
        assert!((out[2] - 0.5).abs() < 5e-3, "center {}", out[2]);
        assert!((out[3] - 0.25).abs() < 5e-3, "right {}", out[3]);
    }

    #[test]
    fn gaussian_blur_fractional_radius_interpolates() {
        // A fractional radius must spread strictly between the two neighbouring
        // integer radii — the old `round()` made it stick to one of them, which
        // caused the weak-sharpen dip around 1.2–1.8 MP.
        let (w, h) = (9usize, 1usize);
        let mut src = vec![0.0f32; w * h];
        src[4] = 1.0;
        let spread = |b: &[f32]| -> f32 {
            b.iter()
                .enumerate()
                .map(|(i, v)| (i as f32 - 4.0).abs() * v)
                .sum()
        };
        let b1 = TinyLumaApp::gaussian_blur_map(&src, w, h, 1.0);
        let b2 = TinyLumaApp::gaussian_blur_map(&src, w, h, 2.0);
        let bf = TinyLumaApp::gaussian_blur_map(&src, w, h, 1.5);
        let (s1, sf, s2) = (spread(&b1), spread(&bf), spread(&b2));
        assert!(s1 < sf && sf < s2, "spread {s1} < {sf} < {s2}");
    }

    #[test]
    fn texture_base_is_constant_on_flat_image() {
        // On a flat frame the texture base equals the color itself at any scale.
        let (w, h) = (16usize, 16usize);
        let color: Vec<u8> = (0..w * h).flat_map(|_| [80u8, 120, 160]).collect();
        let luma: Vec<f32> = (0..w * h)
            .map(|_| 0.2126 * 80.0 / 255.0 + 0.7152 * 120.0 / 255.0 + 0.0722 * 160.0 / 255.0)
            .collect();
        for &scale in &[1.0f32, 2.5] {
            let base = TinyLumaApp::compute_texture_base(&color, &luma, w, h, scale);
            let expected = [80.0 / 255.0, 120.0 / 255.0, 160.0 / 255.0];
            for c in 0..3 {
                assert!(
                    (base.data[c] - expected[c]).abs() < 1e-3,
                    "scale {scale} ch {c}: {} vs {}",
                    base.data[c],
                    expected[c]
                );
            }
        }
    }

    /// The cached spatial precomputations do not change the result: the pass with a ready
    /// `SpatialBase` is byte-identical to the pass that computes the base itself.
    #[test]
    fn spatial_base_cache_is_bit_identical() {
        let (w, h) = (128usize, 96usize);
        let color: Vec<u8> = (0..w * h * 3).map(|i| ((i * 7 + i / 11) % 256) as u8).collect();
        let luma = TinyLumaApp::compute_luma_map(&color);
        let (clarity, sw, sh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
        let grain = vec![0.0f32; w * h * 2];

        let mut s = FilterSettings::default();
        s.texture = 40.0;
        s.clarity = 30.0;
        s.sharpen = 60.0;
        s.grain = 20.0;

        let mut direct = vec![0u8; w * h * 3];
        TinyLumaApp::run_spatial_pass(
            &s, &color, &mut direct, &grain, &luma, &clarity, sw, sh, w, h,
        );

        let base = TinyLumaApp::compute_spatial_base(&s, &color, &luma, w, h);
        let mut cached = vec![0u8; w * h * 3];
        TinyLumaApp::run_spatial_pass_with_base(
            &s, &color, &mut cached, &grain, &clarity, sw, sh, &base, w, h,
        );

        assert_eq!(direct, cached, "cached spatial diverged from direct");
    }

    /// A base captured at one effect strength works for any other strength:
    /// the Texture/Sharpen precomputations do not depend on the slider values.
    #[test]
    fn spatial_base_is_strength_independent() {
        let (w, h) = (96usize, 64usize);
        let color: Vec<u8> = (0..w * h * 3).map(|i| ((i * 13 + 5) % 256) as u8).collect();
        let luma = TinyLumaApp::compute_luma_map(&color);
        let (clarity, sw, sh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
        let grain = vec![0.0f32; w * h * 2];

        // The base is captured at one set of strengths and then reused at others.
        let mut base_settings = FilterSettings::default();
        base_settings.texture = 50.0;
        base_settings.sharpen = 50.0;
        let base = TinyLumaApp::compute_spatial_base(&base_settings, &color, &luma, w, h);

        for (tex, sharp) in [(10.0f32, 10.0f32), (80.0, 120.0), (-40.0, 0.0), (0.0, 90.0)] {
            let mut s = FilterSettings::default();
            s.texture = tex;
            s.clarity = 25.0;
            s.sharpen = sharp;
            s.grain = 15.0;

            let mut direct = vec![0u8; w * h * 3];
            TinyLumaApp::run_spatial_pass(
                &s, &color, &mut direct, &grain, &luma, &clarity, sw, sh, w, h,
            );

            let mut cached = vec![0u8; w * h * 3];
            TinyLumaApp::run_spatial_pass_with_base(
                &s, &color, &mut cached, &grain, &clarity, sw, sh, &base, w, h,
            );

            assert_eq!(direct, cached, "tex={tex} sharp={sharp}");
        }
    }

    /// With the noise controls off the spatial pass must not touch a pixel.
    #[test]
    fn noise_identity_is_bit_identical() {
        let (w, h) = (32usize, 24usize);
        let color: Vec<u8> = (0..w * h * 3).map(|i| ((i * 11 + 3) % 256) as u8).collect();
        let luma = TinyLumaApp::compute_luma_map(&color);
        let (clarity, sw, sh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
        let grain = vec![0.0f32; w * h * 2];
        let s = FilterSettings::default();
        let mut out = vec![0u8; color.len()];
        TinyLumaApp::run_spatial_pass(&s, &color, &mut out, &grain, &luma, &clarity, sw, sh, w, h);
        assert_eq!(out, color, "identity noise settings changed pixels");
    }

    /// Chroma NR must reduce colour speckle on a flat patch.
    #[test]
    fn chroma_noise_reduction_reduces_colour_speckle() {
        let (w, h) = (64usize, 64usize);
        // Flat grey with alternating blue speckle (chroma noise).
        let mut color = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let speck = if (x + y) % 2 == 0 { 12.0f32 } else { -12.0f32 };
                color.push(128);
                color.push(128);
                color.push((128.0 + speck).clamp(0.0, 255.0) as u8);
            }
        }
        let luma = TinyLumaApp::compute_luma_map(&color);
        let (clarity, sw, sh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
        let grain = vec![0.0f32; w * h * 2];

        let mut s = FilterSettings::default();
        s.noise.chroma = 100.0;
        s.noise.detail = 50.0;
        let mut out = vec![0u8; w * h * 3];
        TinyLumaApp::run_spatial_pass(&s, &color, &mut out, &grain, &luma, &clarity, sw, sh, w, h);

        let chroma_var = |buf: &[u8]| -> f32 {
            let n = (w * h) as f32;
            let v: Vec<f32> = buf.chunks(3).map(|p| p[2] as f32 - p[0] as f32).collect();
            let mean = v.iter().sum::<f32>() / n;
            v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n
        };
        let before = chroma_var(&color);
        let after = chroma_var(&out);
        assert!(
            after < before * 0.5,
            "chroma variance did not halve: {before} -> {after}"
        );
    }

    /// Luma NR must smooth flat noise while keeping a hard edge intact.
    #[test]
    fn luma_noise_reduction_smooths_noise_and_keeps_edges() {
        let (w, h) = (48usize, 48usize);
        let mut color = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                let base = if x < w / 2 {
                    // Deterministic ±5 noise on the flat left half.
                    let n = ((y.wrapping_mul(2_654_435_761) ^ x.wrapping_mul(40_503)) >> 16) % 11;
                    (100i32 - 5 + n as i32).clamp(0, 255) as u8
                } else {
                    156
                };
                color[i] = base;
                color[i + 1] = base;
                color[i + 2] = base;
            }
        }
        let luma = TinyLumaApp::compute_luma_map(&color);
        let (clarity, sw, sh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
        let grain = vec![0.0f32; w * h * 2];

        let mut s = FilterSettings::default();
        s.noise.luma = 100.0;
        s.noise.detail = 50.0;
        let mut out = vec![0u8; w * h * 3];
        TinyLumaApp::run_spatial_pass(&s, &color, &mut out, &grain, &luma, &clarity, sw, sh, w, h);

        // Variance of the flat noisy half (columns 4..w/2-4, avoiding the edge).
        let left_var = |buf: &[u8]| -> f32 {
            let mut v = Vec::new();
            for y in 0..h {
                for x in 4..w / 2 - 4 {
                    v.push(buf[(y * w + x) * 3] as f32);
                }
            }
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32
        };
        let before = left_var(&color);
        let after = left_var(&out);
        assert!(
            after < before * 0.6,
            "luma noise not reduced: variance {before} -> {after}"
        );

        // Edge contrast across the middle (rows away from the borders).
        let px = |buf: &[u8], x: usize, y: usize| buf[(y * w + x) * 3] as i32;
        let contrast = px(&out, w - 1, h / 2) - px(&out, 0, h / 2);
        assert!(contrast > 40, "edge contrast lost: {contrast}");
    }

    /// Sharpen masking must not amplify noise on a flat patch.
    #[test]
    fn detail_masking_suppresses_flat_noise() {
        let (w, h) = (48usize, 48usize);
        let mut color = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let n = ((y.wrapping_mul(2_654_435_761) ^ x.wrapping_mul(40_503)) >> 16) % 11;
                let v = (120i32 - 5 + n as i32).clamp(0, 255) as u8;
                let i = (y * w + x) * 3;
                color[i] = v;
                color[i + 1] = v;
                color[i + 2] = v;
            }
        }
        let luma = TinyLumaApp::compute_luma_map(&color);
        let (clarity, sw, sh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
        let grain = vec![0.0f32; w * h * 2];

        let run = |masking: f32| -> Vec<u8> {
            let mut s = FilterSettings::default();
            s.sharpen = 100.0;
            s.detail_masking = masking;
            let mut out = vec![0u8; color.len()];
            TinyLumaApp::run_spatial_pass(
                &s, &color, &mut out, &grain, &luma, &clarity, sw, sh, w, h,
            );
            out
        };
        let variance = |buf: &[u8]| -> f32 {
            let v: Vec<f32> = buf.chunks(3).map(|p| p[0] as f32).collect();
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32
        };

        let unmasked = run(0.0);
        let masked = run(100.0);
        assert!(
            variance(&masked) < variance(&unmasked) * 0.6,
            "masking did not suppress noise: {} -> {}",
            variance(&unmasked),
            variance(&masked)
        );
    }

    /// Sharpen masking must still sharpen a real edge.
    #[test]
    fn detail_masking_keeps_edges() {
        let (w, h) = (48usize, 48usize);
        let mut color = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { 100u8 } else { 156u8 };
                let i = (y * w + x) * 3;
                color[i] = v;
                color[i + 1] = v;
                color[i + 2] = v;
            }
        }
        let luma = TinyLumaApp::compute_luma_map(&color);
        let (clarity, sw, sh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
        let grain = vec![0.0f32; w * h * 2];

        let mut s = FilterSettings::default();
        s.sharpen = 100.0;
        s.detail_masking = 100.0;
        let mut out = vec![0u8; color.len()];
        TinyLumaApp::run_spatial_pass(&s, &color, &mut out, &grain, &luma, &clarity, sw, sh, w, h);

        let px = |buf: &[u8], x: usize, y: usize| buf[(y * w + x) * 3] as i32;
        let input_contrast = 156 - 100;
        let contrast = px(&out, w / 2, h / 2) - px(&out, w / 2 - 1, h / 2);
        assert!(
            contrast > input_contrast + 2,
            "edge not sharpened under masking: {contrast} vs {input_contrast}"
        );
    }

    /// The same shared mask tames the positive Texture boost on flat noise.
    #[test]
    fn detail_masking_suppresses_texture_noise() {
        let (w, h) = (48usize, 48usize);
        let mut color = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let n = ((y.wrapping_mul(2_654_435_761) ^ x.wrapping_mul(40_503)) >> 16) % 11;
                let v = (120i32 - 5 + n as i32).clamp(0, 255) as u8;
                let i = (y * w + x) * 3;
                color[i] = v;
                color[i + 1] = v;
                color[i + 2] = v;
            }
        }
        let luma = TinyLumaApp::compute_luma_map(&color);
        let (clarity, sw, sh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
        let grain = vec![0.0f32; w * h * 2];

        let run = |masking: f32| -> Vec<u8> {
            let mut s = FilterSettings::default();
            s.texture = 100.0;
            s.detail_masking = masking;
            let mut out = vec![0u8; color.len()];
            TinyLumaApp::run_spatial_pass(
                &s, &color, &mut out, &grain, &luma, &clarity, sw, sh, w, h,
            );
            out
        };
        let variance = |buf: &[u8]| -> f32 {
            let v: Vec<f32> = buf.chunks(3).map(|p| p[0] as f32).collect();
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32
        };

        let unmasked = run(0.0);
        let masked = run(100.0);
        assert!(
            variance(&masked) < variance(&unmasked) * 0.6,
            "masking did not tame Texture noise: {} -> {}",
            variance(&unmasked),
            variance(&masked)
        );
    }

    /// Spatial pass budget (Sharpen+Texture+Clarity) at different
    /// resolutions — a check that radius normalization did not kill performance.
    /// `cargo test --release -- --ignored bench_spatial_pass --nocapture`
    #[test]
    #[ignore]
    fn bench_spatial_pass() {
        use std::time::Instant;
        for (w, h) in [(1200usize, 800usize), (4000, 3000)] {
            let n = w * h;
            let color: Vec<u8> = (0..n * 3).map(|i| ((i * 7 + i / 13) % 256) as u8).collect();
            let luma = TinyLumaApp::compute_luma_map(&color);
            let (clarity, csw, csh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
            let grain = vec![0.0f32; n * 2];
            let mut out = vec![0u8; n * 3];
            let mut s = FilterSettings::default();
            s.texture = 50.0;
            s.clarity = 50.0;
            s.sharpen = 50.0;
            let t = Instant::now();
            TinyLumaApp::run_spatial_pass(
                &s, &color, &mut out, &grain, &luma, &clarity, csw, csh, w, h,
            );
            println!(
                "spatial {}x{}: {:.1} ms",
                w,
                h,
                t.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    /// LUT bake budget: how long `run_color_pass` takes on the identity grid
    /// with aggressive edits (many out-of-gamut pixels through gamut_map).
    /// Run `cargo test --release -- --ignored bench_bake_lut --nocapture`.
    #[test]
    #[ignore]
    fn bench_bake_lut() {
        use std::time::Instant;
        let mut s = FilterSettings::default();
        s.saturation = 60.0;
        s.vibrance = 40.0;
        s.temp = 30.0;
        s.contrast = 40.0;
        for size in [33usize, 64] {
            let total = size * size * size;
            let mut buf = vec![0u8; total * 3];
            for iz in 0..size {
                for iy in 0..size {
                    for ix in 0..size {
                        let idx = (ix + iy * size + iz * size * size) * 3;
                        buf[idx] = (ix as f32 / (size - 1) as f32 * 255.0) as u8;
                        buf[idx + 1] = (iy as f32 / (size - 1) as f32 * 255.0) as u8;
                        buf[idx + 2] = (iz as f32 / (size - 1) as f32 * 255.0) as u8;
                    }
                }
            }
            let t = Instant::now();
            TinyLumaApp::run_color_pass(&s, None, &mut buf);
            println!(
                "bake {}^3 ({} points): {:.2} ms",
                size,
                total,
                t.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    /// A non-empty retouch layer must change the full render, and the change
    /// must be confined to the spot's neighbourhood.
    #[test]
    fn render_full_image_applies_retouch_layer() {
        use crate::retouch::{RetouchLayer, Spot, SpotKind};

        let (w, h) = (96u32, 96u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = ((x as f32 / w as f32) * 200.0) as u8;
                img.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        for y in 44..52 {
            for x in 44..52 {
                img.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }

        let mut layer = RetouchLayer::default();
        layer.push(Spot {
            center: [0.5, 0.5],
            radius: 0.1,
            hardness: 1.0,
            opacity: 1.0,
            kind: SpotKind::ProximityMatch,
            path: None,
        });

        let settings = FilterSettings::default();
        let without = TinyLumaApp::render_full_image(&img, &settings, &None, &RetouchLayer::default());
        let with = TinyLumaApp::render_full_image(&img, &settings, &None, &layer);

        assert_ne!(without, with, "the layer must change the render");

        // A far corner must stay byte-identical.
        let far = (2 * 96 + 2) * 3;
        assert_eq!(&without[far..far + 3], &with[far..far + 3]);
        // The hole center must differ.
        let center = (48 * 96 + 48) * 3;
        assert_ne!(&without[center..center + 3], &with[center..center + 3]);
    }

    /// The staged detail + effects passes are byte-identical to the
    /// monolithic `run_spatial_pass_with_base`. Every effect is engaged so the
    /// f32 detail head / effects tail split is exercised end to end.
    #[test]
    fn spatial_stages_match_monolithic() {
        let (w, h) = (96usize, 72usize);
        // Gradient + a blown highlight so Glow / Halation / Flare find energy.
        let mut color = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                let v = (x as f32 / w as f32 * 200.0) as u8;
                color[i] = v;
                color[i + 1] = (v as f32 * 0.8) as u8;
                color[i + 2] = 128;
            }
        }
        for y in 30..40 {
            for x in 40..52 {
                let i = (y * w + x) * 3;
                color[i] = 255;
                color[i + 1] = 255;
                color[i + 2] = 255;
            }
        }

        // All sliders engaged (worst case) — texture/clarity/structure/sharpen,
        // noise reduction, the creative group, grain and vignette.
        let mut s = FilterSettings::default();
        s.texture = 40.0;
        s.clarity = 30.0;
        s.structure = 35.0;
        s.sharpen = 45.0;
        s.detail_masking = 50.0;
        s.noise.luma = 30.0;
        s.noise.chroma = 20.0;
        s.glow = 30.0;
        s.halation = 25.0;
        s.flare = 20.0;
        s.grain = 30.0;
        s.grain_roughness = 40.0;
        s.vignette.amount = -40.0;

        let luma = TinyLumaApp::compute_luma_map(&color);
        let (clarity, csw, csh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
        let grain = TinyLumaApp::generate_grain_map(w, h, 42, s.grain_size, true);
        let base = TinyLumaApp::compute_spatial_base(&s, &color, &luma, w, h);

        let mut mono = vec![0u8; color.len()];
        TinyLumaApp::run_spatial_pass_with_base(
            &s, &color, &mut mono, &grain, &clarity, csw, csh, &base, w, h,
        );

        let mut detail = vec![0.0f32; w * h * 3];
        TinyLumaApp::spatial_detail_pass(
            &s, &color, &clarity, csw, csh, &base, w, h, &mut detail,
        );
        let mut staged = vec![0u8; color.len()];
        TinyLumaApp::spatial_effects_pass(&s, &detail, &grain, &base, w, h, &mut staged);

        assert_eq!(
            mono, staged,
            "staged spatial passes diverged from the monolithic pass"
        );
    }

    /// The detail group's *working grids* are pinned to the 1200 px
    /// reference. At or above the reference the downsample lands on exactly 1200;
    /// below it the grid is the (smaller) proxy — where `detail_radius_factor`
    /// shrinks the fixed radii so the relative effect still matches. This is the
    /// invariant `ARCHITECTURE.md` calls out; without it a 640/960 proxy would
    /// over-strengthen Texture/Clarity/Sharpen versus the full render.
    #[test]
    fn detail_base_grid_is_reference_invariant() {
        let (src_w, src_h) = (2400usize, 1600usize);
        let mut color = vec![0u8; src_w * src_h * 3];
        for y in 0..src_h {
            for x in 0..src_w {
                let i = (y * src_w + x) * 3;
                let c = if ((x / 4) + (y / 4)) % 2 == 0 { 220u8 } else { 40u8 };
                let g = (x as f32 / src_w as f32 * 60.0) as u8;
                let v = c.saturating_add(g);
                color[i] = v;
                color[i + 1] = v;
                color[i + 2] = v;
            }
        }
        let mut s = FilterSettings::default();
        s.texture = 40.0;
        s.sharpen = 40.0;
        s.detail_masking = 40.0;

        for &bucket in &[640usize, 960, 1200, 1920] {
            let (w, h) = TinyLumaApp::proxy_dims_for_long(src_w, src_h, bucket);
            let small = TinyLumaApp::downscale_rgb(&color, src_w, src_h, w, h);
            let luma = TinyLumaApp::compute_luma_map(&small);
            let base = TinyLumaApp::compute_spatial_base(&s, &small, &luma, w, h);
            let tex = base.texture_base.as_ref().expect("texture base present");
            assert_eq!(
                tex.sw.max(tex.sh),
                bucket.min(1200),
                "detail grid not reference-normalized for bucket {bucket}"
            );
            if let Some(m) = &base.mask_base {
                assert_eq!(
                    m.sw.max(m.sh),
                    bucket.min(1200),
                    "mask grid not reference-normalized for bucket {bucket}"
                );
            }
        }
    }

    /// The relative Texture+Sharpen boost stays close across proxy
    /// buckets around the 1200 px reference — the end-to-end guard for the radius
    /// normalization (an un-normalized sub-reference proxy boosts much more).
    #[test]
    fn spatial_effect_is_close_across_proxy_buckets() {
        let (src_w, src_h) = (1200usize, 800usize);
        let mut color = vec![0u8; src_w * src_h * 3];
        for y in 0..src_h {
            for x in 0..src_w {
                let i = (y * src_w + x) * 3;
                let c = if ((x / 3) + (y / 3)) % 2 == 0 { 200u8 } else { 60u8 };
                let g = (y as f32 / src_h as f32 * 50.0) as u8;
                let v = c.saturating_add(g);
                color[i] = v;
                color[i + 1] = v;
                color[i + 2] = v;
            }
        }
        let mut s = FilterSettings::default();
        s.texture = 60.0;
        s.sharpen = 60.0;

        // Mean horizontal luma gradient — a proxy for fine-detail energy.
        let gradient = |rgb: &[u8], w: usize, h: usize| -> f64 {
            let luma = TinyLumaApp::compute_luma_map(rgb);
            let mut sum = 0.0f64;
            for y in 0..h {
                for x in 0..w.saturating_sub(1) {
                    sum += (luma[y * w + x + 1] - luma[y * w + x]).abs() as f64;
                }
            }
            sum / (w * h) as f64
        };

        let mut ratios = Vec::new();
        for &bucket in &[640usize, 960, 1200] {
            let (w, h) = TinyLumaApp::proxy_dims_for_long(src_w, src_h, bucket);
            let small = TinyLumaApp::downscale_rgb(&color, src_w, src_h, w, h);
            let luma = TinyLumaApp::compute_luma_map(&small);
            let mut out = vec![0u8; small.len()];
            TinyLumaApp::run_spatial_pass(
                &s, &small, &mut out, &[], &luma, &[], 0, 0, w, h,
            );
            ratios.push(gradient(&out, w, h) / gradient(&small, w, h).max(1e-9));
        }
        let ref_ratio = ratios[2]; // bucket 1200
        for (i, r) in ratios.iter().enumerate() {
            assert!(
                *r / ref_ratio > 0.6 && *r / ref_ratio < 1.6,
                "relative detail boost drifted across proxy buckets: {ratios:?} (idx {i})"
            );
        }
    }

    /// The export path (`render_full_image`) is independent of the
    /// interactive proxy — it renders at the full source resolution, never capped
    /// to a proxy bucket. Guards against a future proxy leaking into export.
    #[test]
    fn export_renders_at_full_resolution_not_the_proxy() {
        let (w, h) = (1500usize, 1000usize); // larger than the 1200 preview cap
        let mut img = RgbImage::new(w as u32, h as u32);
        for (x, _y, p) in img.enumerate_pixels_mut() {
            let v = ((x % 64) as f32 / 64.0 * 255.0) as u8;
            *p = image::Rgb([v, v / 2, 200]);
        }
        let mut s = FilterSettings::default();
        s.texture = 30.0; // a spatial effect, so the full-res pass runs

        let out = TinyLumaApp::render_full_image(
            &img,
            &s,
            &None,
            &crate::retouch::RetouchLayer::default(),
        );
        assert_eq!(out.len(), w * h * 3, "export must keep the full resolution");
        assert_ne!(
            out.as_slice(),
            img.as_raw().as_slice(),
            "export must apply the settings at full resolution"
        );
    }

    /// The drag tier caps the long side and preserves the aspect ratio.
    #[test]
    fn drag_dims_caps_and_preserves_aspect() {
        assert_eq!(TinyLumaApp::drag_dims(1200, 800), (640, 427));
        assert_eq!(TinyLumaApp::drag_dims(800, 600), (640, 480));
        // At or below the cap the size is unchanged (no upscale).
        assert_eq!(TinyLumaApp::drag_dims(640, 480), (640, 480));
        assert_eq!(TinyLumaApp::drag_dims(320, 240), (320, 240));
    }

    /// The drag-tier settings signatures split detail from base — a
    /// detail change moves the detail signature only, and an effect *amount*
    /// change moves neither (the effects tail always re-runs).
    #[test]
    fn drag_signatures_track_the_right_settings() {
        let mut a = FilterSettings::default();
        a.texture = 40.0;
        a.glow = 30.0;

        let b = a;
        assert_eq!(
            TinyLumaApp::drag_detail_signature(&a),
            TinyLumaApp::drag_detail_signature(&b)
        );
        assert_eq!(
            TinyLumaApp::drag_base_signature(&a, 640, 427),
            TinyLumaApp::drag_base_signature(&b, 640, 427)
        );

        // A detail slider change moves the detail signature…
        let mut b2 = a;
        b2.texture = 41.0;
        assert_ne!(
            TinyLumaApp::drag_detail_signature(&a),
            TinyLumaApp::drag_detail_signature(&b2)
        );
        // …but not the base signature (engagement unchanged).
        assert_eq!(
            TinyLumaApp::drag_base_signature(&a, 640, 427),
            TinyLumaApp::drag_base_signature(&b2, 640, 427)
        );

        // Tuning an effect amount moves neither signature.
        let mut c = a;
        c.glow = 55.0;
        assert_eq!(
            TinyLumaApp::drag_detail_signature(&a),
            TinyLumaApp::drag_detail_signature(&c)
        );
        assert_eq!(
            TinyLumaApp::drag_base_signature(&a, 640, 427),
            TinyLumaApp::drag_base_signature(&c, 640, 427)
        );

        // Engaging an effect, or a different resolution, moves the base signature.
        let mut d = a;
        d.halation = 25.0;
        assert_ne!(
            TinyLumaApp::drag_base_signature(&a, 640, 427),
            TinyLumaApp::drag_base_signature(&d, 640, 427)
        );
        assert_ne!(
            TinyLumaApp::drag_base_signature(&a, 640, 427),
            TinyLumaApp::drag_base_signature(&a, 800, 533)
        );
    }

    /// The drag-tier downscale averages (identity when unchanged).
    #[test]
    fn drag_downscale_matches_average() {
        // 2x2 → 1x1 averages the four red samples.
        let src = vec![0u8, 0, 0, 100, 0, 0, 200, 0, 0, 40, 0, 0];
        let out = TinyLumaApp::downscale_rgb(&src, 2, 2, 1, 1);
        assert_eq!(out, vec![85u8, 0, 0]); // (0 + 100 + 200 + 40) / 4

        // Identity when the dimensions are unchanged.
        let same = TinyLumaApp::downscale_rgb(&src, 2, 2, 2, 2);
        assert_eq!(same, src);
    }

    /// The proxy long side snaps up to the ladder, is floored at the
    /// minimum, and caps at the top rung without needing to know the source.
    #[test]
    fn proxy_target_long_snaps_to_ladder() {
        // Below the floor → the first rung.
        assert_eq!(TinyLumaApp::proxy_target_long(100.0, 0.5, 1.0), 640);
        // Just above a rung → the next rung.
        assert_eq!(TinyLumaApp::proxy_target_long(1000.0, 1.0, 1.0), 1280);
        // Exactly on a rung → that rung.
        assert_eq!(TinyLumaApp::proxy_target_long(960.0, 1.0, 1.0), 960);
        // HiDPI raises the target.
        assert_eq!(TinyLumaApp::proxy_target_long(1000.0, 1.0, 1.5), 1920);
        // Above the top rung → the top rung.
        assert_eq!(TinyLumaApp::proxy_target_long(4000.0, 5.0, 1.0), 4096);
    }

    /// `proxy_dims` never upscales beyond the source, preserves the
    /// aspect ratio, and tracks viewport / zoom / pixel density.
    #[test]
    fn proxy_dims_caps_to_source_and_preserves_aspect() {
        // Source at/below the fit level is returned unchanged.
        assert_eq!(
            TinyLumaApp::proxy_dims(1200, 800, 1000.0, 1.0, 1.0),
            (1200, 800)
        );
        // A large source is downscaled to the viewport bucket (aspect kept).
        assert_eq!(
            TinyLumaApp::proxy_dims(4000, 3000, 1000.0, 1.0, 1.0),
            (1280, 960)
        );
        // HiDPI raises the bucket.
        assert_eq!(
            TinyLumaApp::proxy_dims(4000, 3000, 1000.0, 1.0, 1.5),
            (1920, 1440)
        );
        // Zooming to 1:1 uses the full source.
        assert_eq!(
            TinyLumaApp::proxy_dims(4000, 3000, 1000.0, 4.0, 1.0),
            (4000, 3000)
        );
        // Tiny sources are never upscaled.
        assert_eq!(
            TinyLumaApp::proxy_dims(320, 240, 1000.0, 1.0, 1.0),
            (320, 240)
        );
        // A huge source caps at the top rung, not the huge native size.
        assert_eq!(
            TinyLumaApp::proxy_dims(8000, 6000, 2000.0, 5.0, 1.0),
            (4096, 3072)
        );
    }
}
