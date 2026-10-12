// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! Benchmarks and performance tests for frame processing.
//!
//! Scenario: build "reference" settings with ALL sliders, an active
//! external LUT and all spatial effects, apply them to a
//! synthetic preview frame, then "virtually" nudge each slider
//! individually and measure the cost.
//!
//! The model mirrors the incremental logic of [`TinyLumaApp::process_preview`]:
//! - color sliders → LUT bake (33³ during drag / 64³ on release)
//!   + application to the buffer; on release additionally the spatial chain;
//! - spatial (texture/clarity/sharpen/grain) → `run_spatial_pass`
//!   on top of the already-prepared luma/clarity/grain caches;
//! - dehaze → recompute dehaze + luma/clarity maps + `run_spatial_pass`.
//!
//! Measurements (in release, otherwise the numbers are not meaningful):
//! `cargo test --release -- --ignored bench_sliders_preview --nocapture`
//! `cargo test --release -- --ignored bench_sliders_hires --nocapture`

use std::time::{Duration, Instant};

use eframe::egui;
use rayon::prelude::*;

use super::TinyLumaApp;
use super::processing::SpatialBase;
use crate::lut::Lut3D;
use crate::settings::FilterSettings;

/// Step of the "virtual" slider movement for the -100..100 / 0..100 ranges.
const NUDGE: f32 = 5.0;
/// How many times to repeat each measurement (averaged to smooth out noise).
const REPEATS: usize = 7;

// ─────────────────────────────────────────────────────────────────────────────
// Sliders
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Slider {
    Exposure,
    Contrast,
    Whites,
    Blacks,
    Shadows,
    Highlights,
    Temp,
    Tint,
    Vibrance,
    Saturation,
    Texture,
    Clarity,
    Sharpen,
    Dehaze,
    Grain,
    GrainSize,
    LutIntensity,
    MixerSat,
    GradingSat,
    Brightness,
    Hue,
    VignetteAmount,
    GrainRoughness,
    Structure,
    Glow,
    Halation,
    Flare,
}

impl Slider {
    const ALL: [Slider; 27] = [
        Slider::Exposure,
        Slider::Contrast,
        Slider::Whites,
        Slider::Blacks,
        Slider::Shadows,
        Slider::Highlights,
        Slider::Temp,
        Slider::Tint,
        Slider::Vibrance,
        Slider::Saturation,
        Slider::Texture,
        Slider::Clarity,
        Slider::Sharpen,
        Slider::Dehaze,
        Slider::Grain,
        Slider::GrainSize,
        Slider::LutIntensity,
        Slider::MixerSat,
        Slider::GradingSat,
        Slider::Brightness,
        Slider::Hue,
        Slider::VignetteAmount,
        Slider::GrainRoughness,
        Slider::Structure,
        Slider::Glow,
        Slider::Halation,
        Slider::Flare,
    ];

    fn name(self) -> &'static str {
        match self {
            Slider::Exposure => "Exposure",
            Slider::Contrast => "Contrast",
            Slider::Whites => "Whites",
            Slider::Blacks => "Blacks",
            Slider::Shadows => "Shadows",
            Slider::Highlights => "Highlights",
            Slider::Temp => "Temp",
            Slider::Tint => "Tint",
            Slider::Vibrance => "Vibrance",
            Slider::Saturation => "Saturation",
            Slider::Texture => "Texture",
            Slider::Clarity => "Clarity",
            Slider::Sharpen => "Sharpen",
            Slider::Dehaze => "Dehaze",
            Slider::Grain => "Grain",
            Slider::GrainSize => "GrainSize",
            Slider::LutIntensity => "LutIntensity",
            Slider::MixerSat => "MixerSat",
            Slider::GradingSat => "GradingSat",
            Slider::Brightness => "Brightness",
            Slider::Hue => "Hue",
            Slider::VignetteAmount => "VignetteAmount",
            Slider::GrainRoughness => "GrainRoughness",
            Slider::Structure => "Structure",
            Slider::Glow => "Glow",
            Slider::Halation => "Halation",
            Slider::Flare => "Flare",
        }
    }

    /// Sliders that are baked into the color LUT (per-pixel, no spatial).
    fn is_color(self) -> bool {
        matches!(
            self,
            Slider::Exposure
                | Slider::Contrast
                | Slider::Whites
                | Slider::Blacks
                | Slider::Shadows
                | Slider::Highlights
                | Slider::Temp
                | Slider::Tint
                | Slider::Vibrance
                | Slider::Saturation
                | Slider::LutIntensity
                | Slider::MixerSat
                | Slider::GradingSat
                | Slider::Brightness
                | Slider::Hue
        )
    }

    /// Nudges the slider by a small step (staying within its range).
    fn nudge(self, s: &mut FilterSettings) {
        match self {
            Slider::Exposure => s.exposure += 0.5,
            Slider::Contrast => s.contrast += NUDGE,
            Slider::Whites => s.whites += NUDGE,
            Slider::Blacks => s.blacks += NUDGE,
            Slider::Shadows => s.shadows += NUDGE,
            Slider::Highlights => s.highlights += NUDGE,
            Slider::Temp => s.temp += NUDGE,
            Slider::Tint => s.tint += NUDGE,
            Slider::Vibrance => s.vibrance += NUDGE,
            Slider::Saturation => s.saturation += NUDGE,
            Slider::Texture => s.texture += NUDGE,
            Slider::Clarity => s.clarity += NUDGE,
            Slider::Sharpen => s.sharpen += 7.5,
            Slider::Dehaze => s.dehaze += NUDGE,
            Slider::Grain => s.grain += NUDGE,
            Slider::GrainSize => s.grain_size += NUDGE,
            Slider::LutIntensity => s.lut_intensity += NUDGE,
            Slider::MixerSat => s.hsl.bands[0][1] += NUDGE,
            Slider::GradingSat => {
                s.grading.shadows.sat += NUDGE;
                s.grading.shadows.hue = 30.0;
            }
            Slider::Brightness => s.brightness += NUDGE,
            Slider::Hue => s.hue += NUDGE,
            Slider::VignetteAmount => s.vignette.amount += NUDGE,
            Slider::GrainRoughness => s.grain_roughness += NUDGE,
            Slider::Structure => s.structure += NUDGE,
            Slider::Glow => s.glow += NUDGE,
            Slider::Halation => s.halation += NUDGE,
            Slider::Flare => s.flare += NUDGE,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference data
// ─────────────────────────────────────────────────────────────────────────────

/// Settings where ALL sliders are engaged (none at default).
fn baseline_settings() -> FilterSettings {
    FilterSettings {
        exposure: 0.6,
        contrast: 25.0,
        whites: 18.0,
        blacks: -12.0,
        shadows: 30.0,
        highlights: -25.0,
        temp: 15.0,
        tint: -10.0,
        vibrance: 30.0,
        saturation: 12.0,
        texture: 35.0,
        clarity: 28.0,
        dehaze: 18.0,
        sharpen: 45.0,
        lut_intensity: 85.0,
        grain: 30.0,
        grain_size: 60.0,
        grain_roughness: 40.0,
        brightness: 20.0,
        hue: 40.0,
        structure: 35.0,
        glow: 30.0,
        halation: 25.0,
        flare: 20.0,
        vignette: crate::pipeline::vignette::VignetteSettings {
            amount: -40.0,
            ..Default::default()
        },
        grading: crate::pipeline::grading::ColorGrading {
            shadows: crate::pipeline::grading::GradingWheel {
                hue: 210.0,
                sat: 40.0,
                lum: 0.0,
            },
            ..Default::default()
        },
        ..FilterSettings::default()
    }
}

/// Synthetic "photographic" frame: gradient + low-frequency checkerboard
/// (provides edges for texture/clarity/sharpen/dehaze) + fine noise.
fn synth_image(w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let check = if ((x / 32) + (y / 32)) % 2 == 0 {
                1.0
            } else {
                0.35
            };
            let noise = ((x.wrapping_mul(2_654_435_761) ^ y.wrapping_mul(40_503)) >> 13) & 0x3f;
            let base = (fx * 0.6 + fy * 0.4) * check;
            let r = ((base * 255.0) as i32 + noise as i32).clamp(0, 255) as u8;
            let g = ((base * 0.8 * 255.0) as i32 + noise as i32).clamp(0, 255) as u8;
            let b = ((base * 0.6 * 255.0) as i32 + noise as i32).clamp(0, 255) as u8;
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

/// Synthetic external LUT: a soft S-curve per channel. Not identity,
/// so `lut.apply` does real work (trilinear interpolation).
fn synth_lut(size: usize) -> Lut3D {
    let s = (size - 1).max(1) as f32;
    let smooth = |v: f32| v * v * (3.0 - 2.0 * v);
    let mut data = Vec::with_capacity(size * size * size);
    for iz in 0..size {
        for iy in 0..size {
            for ix in 0..size {
                data.push([
                    smooth(ix as f32 / s),
                    smooth(iy as f32 / s),
                    smooth(iz as f32 / s),
                ]);
            }
        }
    }
    Lut3D { size, data }
}

/// Smooth gradient (R along X, G along Y, B — a blend) — most sensitive to
/// banding from the LUT's trilinear interpolation.
fn smooth_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / (w - 1).max(1) as f32;
            let fy = y as f32 / (h - 1).max(1) as f32;
            out.push((fx * 255.0).round() as u8);
            out.push((fy * 255.0).round() as u8);
            out.push(((fx + fy) * 0.5 * 255.0).round() as u8);
        }
    }
    out
}

/// Applies the baked LUT to an RGB buffer (as in `process_preview`).
fn apply_lut(lut: &Lut3D, buf: &mut [u8]) {
    buf.par_chunks_mut(3).for_each(|p| {
        let r = p[0] as f32 / 255.0;
        let g = p[1] as f32 / 255.0;
        let b = p[2] as f32 / 255.0;
        let m = lut.apply(r, g, b);
        p[0] = (m[0] * 255.0).clamp(0.0, 255.0) as u8;
        p[1] = (m[1] * 255.0).clamp(0.0, 255.0) as u8;
        p[2] = (m[2] * 255.0).clamp(0.0, 255.0) as u8;
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Harness
// ─────────────────────────────────────────────────────────────────────────────

/// State replicating `TinyLumaApp`'s buffers/caches for a single frame.
struct Harness {
    w: usize,
    h: usize,
    base: Vec<u8>,    // input (downscaled preview frame)
    color: Vec<u8>,   // color_buffer (after the LUT)
    dehazed: Vec<u8>, // dehazed_buffer
    output: Vec<u8>,  // processed_pixels
    luma: Vec<f32>,
    clarity: Vec<f32>,
    clarity_sw: usize,
    clarity_sh: usize,
    spatial_base: SpatialBase,
    grain: Vec<f32>,
    /// Grain-size slider value `grain` was generated for.
    grain_size: f32,
    /// Whether `grain` carries a rough channel (Grain Roughness engaged).
    grain_rough: bool,
    lut: Lut3D,
    combined: Lut3D,
    settings: FilterSettings,
    baseline: FilterSettings,
}

impl Harness {
    fn new(w: usize, h: usize, baseline: FilterSettings, lut: Lut3D) -> Self {
        let base = synth_image(w, h);
        let n = base.len();
        let grain = TinyLumaApp::generate_grain_map(
            w,
            h,
            42,
            baseline.grain_size,
            baseline.grain_roughness > 0.0,
        );
        Self {
            w,
            h,
            base: base.clone(),
            color: base.clone(),
            dehazed: vec![0u8; n],
            output: vec![0u8; n],
            luma: Vec::new(),
            clarity: Vec::new(),
            clarity_sw: 0,
            clarity_sh: 0,
            spatial_base: SpatialBase::default(),
            grain,
            grain_size: baseline.grain_size,
            grain_rough: baseline.grain_roughness > 0.0,
            lut,
            combined: Lut3D {
                size: 2,
                data: vec![[0.0; 3]; 8],
            },
            settings: baseline,
            baseline,
        }
    }

    /// Return to the reference settings + full cache recompute.
    fn reset(&mut self) {
        self.settings = self.baseline;
        self.recompute_full();
    }

    /// Pass 1: bake a LUT of the given size and apply it to the buffer.
    fn apply_color(&mut self, size: usize) {
        self.combined = TinyLumaApp::bake_lut(&self.settings, Some(&self.lut), size);
        self.color.copy_from_slice(&self.base);
        apply_lut(&self.combined, &mut self.color);
    }

    /// Applies dehaze to the color buffer (fast — drag quality).
    fn apply_dehaze(&mut self, fast: bool) {
        let use_dehaze = self.settings.dehaze != 0.0;
        if !use_dehaze {
            return;
        }
        if self.dehazed.len() != self.color.len() {
            self.dehazed.resize(self.color.len(), 0);
        }
        if fast {
            crate::pipeline::dehaze::apply_fast(
                &self.color,
                &mut self.dehazed,
                self.w,
                self.h,
                self.settings.dehaze / 100.0,
            );
        } else {
            crate::pipeline::dehaze::apply(
                &self.color,
                &mut self.dehazed,
                self.w,
                self.h,
                self.settings.dehaze / 100.0,
            );
        }
    }

    /// Recompute luma/clarity/spatial_base from the current spatial input.
    fn recompute_caches(&mut self) {
        let use_dehaze = self.settings.dehaze != 0.0;
        let (luma, clarity, sw, sh, base) = {
            let input: &[u8] = if use_dehaze {
                &self.dehazed
            } else {
                &self.color
            };
            let luma = TinyLumaApp::compute_luma_map(input);
            let (clarity, sw, sh) = if self.settings.clarity != 0.0 {
                TinyLumaApp::compute_clarity_cache(&luma, self.w, self.h)
            } else {
                (Vec::new(), 0, 0)
            };
            let base =
                TinyLumaApp::compute_spatial_base(&self.settings, input, &luma, self.w, self.h);
            (luma, clarity, sw, sh, base)
        };
        self.luma = luma;
        self.clarity = clarity;
        self.clarity_sw = sw;
        self.clarity_sh = sh;
        self.spatial_base = base;
    }

    /// Dehaze + cache recompute (as on release).
    fn recompute_dehaze_and_caches(&mut self, fast: bool) {
        self.apply_dehaze(fast);
        self.recompute_caches();
    }

    /// Spatial pass on top of the prepared caches.
    fn run_spatial(&mut self) {
        // Mirror the app: regenerate the grain map when the Size slider moved or
        // the Roughness (on/off) changed.
        let want_rough = self.settings.grain_roughness > 0.0;
        if self.settings.grain != 0.0
            && (self.grain_size != self.settings.grain_size || self.grain_rough != want_rough)
        {
            self.grain = TinyLumaApp::generate_grain_map(
                self.w,
                self.h,
                42,
                self.settings.grain_size,
                want_rough,
            );
            self.grain_size = self.settings.grain_size;
            self.grain_rough = want_rough;
        }
        let use_dehaze = self.settings.dehaze != 0.0;
        let input: &[u8] = if use_dehaze {
            &self.dehazed
        } else {
            &self.color
        };
        let needs_spatial = self.settings.texture != 0.0
            || self.settings.clarity != 0.0
            || self.settings.sharpen > 0.0
            || self.settings.grain != 0.0
            || !self.settings.noise.is_identity()
            || !self.settings.vignette.is_identity();
        if needs_spatial {
            TinyLumaApp::run_spatial_pass_with_base(
                &self.settings,
                input,
                &mut self.output,
                &self.grain,
                &self.clarity,
                self.clarity_sw,
                self.clarity_sh,
                &self.spatial_base,
                self.w,
                self.h,
            );
        } else {
            self.output.copy_from_slice(input);
        }
    }

    /// Cost of one interactive colour-drag frame: 33³ bake + apply + histogram +
    /// the CPU side of the texture upload (building the `ColorImage`). Mirrors
    /// [`TinyLumaApp::process_preview`] with `drag_active && color_dirty` (spatial skipped).
    fn color_drag_frame(&mut self) -> Duration {
        let t = Instant::now();
        self.apply_color(33);
        let _hist = super::histogram::Histogram::of_rgb8(&self.color);
        let _img = egui::ColorImage::from_rgb([self.w, self.h], &self.color);
        t.elapsed()
    }

    /// Cost of one interactive spatial-drag frame: the full spatial loop on the
    /// already-prepared caches + histogram + upload CPU side. Spatial sliders do
    /// NOT invalidate the caches, so this is the per-frame cost the plan targets.
    fn spatial_drag_frame(&mut self) -> Duration {
        let t = Instant::now();
        self.run_spatial();
        let _hist = super::histogram::Histogram::of_rgb8(&self.output);
        let _img = egui::ColorImage::from_rgb([self.w, self.h], &self.output);
        t.elapsed()
    }

    /// Full frame recompute (as in export/slider release).
    /// Full frame recompute (as on release): preview-size LUT,
    /// dehaze + caches + spatial.
    fn recompute_full(&mut self) {
        self.apply_color(33);
        self.recompute_dehaze_and_caches(false);
        self.run_spatial();
    }

    /// Cost of a virtual nudge of a single slider.
    ///
    /// Returns `(drag, full)`:
    /// - `drag` — the cost of a frame during dragging;
    /// - `full` — the cost of a full recompute on release (for spatial sliders
    ///   it equals `drag`, since the LUT is not re-baked).
    fn measure(&mut self, slider: Slider) -> (Duration, Duration) {
        let saved = self.settings;
        slider.nudge(&mut self.settings);

        let (drag, full) = if slider.is_color() {
            // Drag: 33³ LUT + application (spatial is skipped).
            let t = Instant::now();
            self.apply_color(33);
            let drag = t.elapsed();

            // Release: same 33³, but plus dehaze/luma/clarity/spatial.
            let t = Instant::now();
            self.recompute_full();
            let full = t.elapsed();
            (drag, full)
        } else {
            // Spatial slider. For Texture/Clarity/Sharpen/Grain
            // only their strength changes — the caches are left alone. For Dehaze the
            // spatial input changes: drag computes a fast map, release — the full one.
            let dehaze_changed = self.settings.dehaze != saved.dehaze;

            let t = Instant::now();
            if dehaze_changed {
                // Drag: dehaze only, Details caches are NOT touched (they catch up on release).
                self.apply_dehaze(true);
            }
            self.run_spatial();
            let drag = t.elapsed();

            let full = if dehaze_changed {
                let t = Instant::now();
                self.recompute_dehaze_and_caches(false);
                self.run_spatial();
                t.elapsed()
            } else {
                drag
            };
            (drag, full)
        };

        self.settings = saved;
        (drag, full)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Report printing
// ─────────────────────────────────────────────────────────────────────────────

fn bench_at(w: usize, h: usize, repeats: usize) {
    let mut hz = Harness::new(w, h, baseline_settings(), synth_lut(33));
    hz.reset();

    println!("\n=== {w}x{h} ===");
    println!(
        "{:<13} {:>12} {:>12}",
        "Slider", "drag, ms", "full, ms"
    );
    println!("{}", "-".repeat(39));

    let mut worst_drag = (0.0f64, "".to_string());
    let mut worst_full = (0.0f64, "".to_string());

    for s in Slider::ALL {
        let mut drag = 0.0f64;
        let mut full = 0.0f64;
        for _ in 0..repeats {
            // Reset the caches to the reference before each measurement.
            hz.reset();
            let (d, f) = hz.measure(s);
            drag += d.as_secs_f64();
            full += f.as_secs_f64();
        }
        let drag_ms = drag / repeats as f64 * 1000.0;
        let full_ms = full / repeats as f64 * 1000.0;
        println!("{:<13} {:>12.2} {:>12.2}", s.name(), drag_ms, full_ms);

        if drag_ms > worst_drag.0 {
            worst_drag = (drag_ms, s.name().to_string());
        }
        if full_ms > worst_full.0 {
            worst_full = (full_ms, s.name().to_string());
        }
    }

    println!("{}", "-".repeat(39));
    println!(
        "worst drag: {} ({:.2} ms) | worst full: {} ({:.2} ms)",
        worst_drag.1, worst_drag.0, worst_full.1, worst_full.0
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// Harness correctness: every slider actually changes the resulting frame.
/// Runs in a regular `cargo test` (without `--ignored`), on a small frame.
#[test]
fn every_slider_changes_output() {
    let mut hz = Harness::new(320, 240, baseline_settings(), synth_lut(33));
    hz.reset();
    let reference = hz.output.clone();

    for s in Slider::ALL {
        hz.reset();
        s.nudge(&mut hz.settings);
        hz.recompute_full();

        let diff = hz
            .output
            .iter()
            .zip(reference.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            diff > 0,
            "slider {} did not change the frame (diff = 0)",
            s.name()
        );
    }
}

/// Measures the cost of nudging each slider at preview resolution.
/// `cargo test --release -- --ignored bench_sliders_preview --nocapture`
#[test]
#[ignore]
fn bench_sliders_preview() {
    bench_at(1200, 800, REPEATS);
}

/// The same measurement on a full-size frame (closer to export). More expensive.
/// `cargo test --release -- --ignored bench_sliders_hires --nocapture`
#[test]
#[ignore]
fn bench_sliders_hires() {
    bench_at(4000, 3000, 1);
}

/// Interactive drag-frame cost: what the UI thread must do per frame while the
/// user drags a colour vs a spatial slider, compared to a full release render.
/// `cargo test --release -- --ignored bench_interactive_drag_frames --nocapture`
#[test]
#[ignore]
fn bench_interactive_drag_frames() {
    for (w, h) in [(1200usize, 800usize), (4000, 3000)] {
        let reps = if w >= 4000 { 1 } else { REPEATS };
        let mut hz = Harness::new(w, h, baseline_settings(), synth_lut(33));
        hz.reset();

        let mut color_ms = 0.0f64;
        let mut spatial_ms = 0.0f64;
        let mut full_ms = 0.0f64;
        for _ in 0..reps {
            hz.reset();
            color_ms += hz.color_drag_frame().as_secs_f64();
            hz.reset();
            spatial_ms += hz.spatial_drag_frame().as_secs_f64();
            hz.reset();
            let t = Instant::now();
            hz.recompute_full();
            let _ = super::histogram::Histogram::of_rgb8(&hz.output);
            full_ms += t.elapsed().as_secs_f64();
        }
        let n = reps as f64;
        println!("\n=== interactive frames {w}x{h} (avg of {reps}) ===");
        println!("colour drag frame   {:>8.2} ms", color_ms / n * 1000.0);
        println!("spatial drag frame  {:>8.2} ms", spatial_ms / n * 1000.0);
        println!("release full render {:>8.2} ms", full_ms / n * 1000.0);
    }
}

/// The dynamic viewport proxy. For each ladder bucket, the cost of
/// deriving the proxy from the higher-res source (`downscale_rgb`) and of the
/// authoritative render at that resolution, so the per-bucket frame budget is
/// visible when tuning the ladder.
/// `cargo test --release -- --ignored bench_proxy_resolution --nocapture`
#[test]
#[ignore]
fn bench_proxy_resolution() {
    use std::sync::Arc;
    use std::time::Instant;

    use super::proxy_job::{ProxyRequest, ProxyWorker};
    use crate::retouch::RetouchLayer;

    // A DSLR-sized source: the proxy is always a downscale of it.
    let (src_w, src_h) = (4096usize, 3072usize);
    let s = baseline_settings();
    let src = Arc::new(
        image::RgbImage::from_raw(src_w as u32, src_h as u32, synth_image(src_w, src_h))
            .expect("source dims match the buffer"),
    );

    // The UI thread only enqueues; the worker downscales + renders.
    let ctx = egui::Context::default();
    let mut worker = ProxyWorker::spawn(&ctx);

    println!("\n=== proxy resolution (source {src_w}x{src_h}) ===");
    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>14}",
        "bucket", "rebuild ms", "render ms", "total ms", "ui enqueue ms"
    );
    for &bucket in &[640usize, 960, 1280, 1920, 2560] {
        let (w, h) = TinyLumaApp::proxy_dims_for_long(src_w, src_h, bucket);

        // Rebuild: derive the working proxy from the source.
        // Now runs on the worker thread; shown for the per-bucket budget.
        let t = Instant::now();
        let small = TinyLumaApp::downscale_rgb(src.as_raw(), src_w, src_h, w, h);
        let rebuild_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Render: the full spatial chain at the proxy resolution (worker thread).
        let luma = TinyLumaApp::compute_luma_map(&small);
        let (clarity, csw, csh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
        let grain =
            TinyLumaApp::generate_grain_map(w, h, 42, s.grain_size, s.grain_roughness > 0.0);
        let base = TinyLumaApp::compute_spatial_base(&s, &small, &luma, w, h);
        let mut out = vec![0u8; small.len()];
        let t = Instant::now();
        TinyLumaApp::run_spatial_pass_with_base(
            &s, &small, &mut out, &grain, &clarity, csw, csh, &base, w, h,
        );
        let render_ms = t.elapsed().as_secs_f64() * 1000.0;

        // UI-thread cost of a settle: build the request (Arc/settings clones)
        // and enqueue. Must stay far under one frame (<= ~16 ms).
        let t = Instant::now();
        worker.enqueue(ProxyRequest {
            revision: bucket as u64,
            source: src.clone(),
            target: (w, h),
            settings: s,
            lut: None,
            layer: Arc::new(RetouchLayer::default()),
        });
        let enqueue_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!(
            "{:>10} {:>12.2} {:>12.2} {:>12.2} {:>14.4}",
            format!("{w}x{h}"),
            rebuild_ms,
            render_ms,
            rebuild_ms + render_ms,
            enqueue_ms
        );
    }
}

/// Full-resolution heal latency: the synchronous cost the UI thread currently
/// pays on stroke release (`ensure_full_healed` → `apply_spot_cached`), cold vs
/// warm, at the 4096px retouch working size.
/// `cargo test --release -- --ignored bench_heal_latency --nocapture`
#[test]
#[ignore]
fn bench_heal_latency() {
    use crate::retouch::brush::BrushSettings;
    use crate::retouch::spot::{apply_spot_cached, SpotCache};

    let (w, h) = (4096u32, 3072u32);
    let base = image::RgbImage::from_fn(w, h, |x, y| {
        let v = ((x * 7 + y * 3) % 200) as u8;
        image::Rgb([v, 255 - v, v / 2])
    });

    // A medium stroke whose bbox stays under the spot-cache cap, so the warm
    // replay is a genuine cache hit (blit), not a second recompute.
    let brush = BrushSettings {
        size: 80.0,
        hardness: 0.6,
        spacing: 0.25,
        ..Default::default()
    };
    let path = [[0.3f32, 0.4], [0.6, 0.6]];
    let stroke = brush.to_stroke(&path, w as usize, h as usize);

    // Cold: no cache — the real content-aware fill.
    let mut cold = base.clone();
    let t = Instant::now();
    apply_spot_cached(&mut SpotCache::default(), &mut cold, &stroke);
    let cold_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Warm: the same stroke replayed — the memoized region blit.
    let mut cache = SpotCache::default();
    let mut warm = base.clone();
    apply_spot_cached(&mut cache, &mut warm, &stroke);
    let mut warm2 = base.clone();
    let t = Instant::now();
    apply_spot_cached(&mut cache, &mut warm2, &stroke);
    let warm_ms = t.elapsed().as_secs_f64() * 1000.0;

    println!("\n=== heal latency {w}x{h} (stroke size 80) ===");
    println!("cold fill (first heal) {:>8.1} ms", cold_ms);
    println!("warm blit (cache hit)  {:>8.1} ms", warm_ms);
}

/// The heal fill now runs on the background worker. Measures the
/// UI-thread enqueue cost (what a stroke release pays — must be a small
/// fraction of one frame, not the multi-second fill) and the worker's
/// end-to-end rebuild latency, at the 4096px retouch size.
/// `cargo test --release -- --ignored bench_heal_async_offload --nocapture`
#[test]
#[ignore]
fn bench_heal_async_offload() {
    use crate::app::heal_job::{HealReply, HealRequest, HealWorker};
    use crate::retouch::brush::BrushSettings;
    use std::sync::Arc;
    use std::time::Duration;

    let (w, h) = (4096u32, 3072u32);
    let base = image::RgbImage::from_fn(w, h, |x, y| {
        let v = ((x * 7 + y * 3) % 200) as u8;
        image::Rgb([v, 255 - v, v / 2])
    });
    let brush = BrushSettings {
        size: 80.0,
        hardness: 0.6,
        spacing: 0.25,
        ..Default::default()
    };
    let stroke = brush.to_stroke(&[[0.3f32, 0.4], [0.6, 0.6]], w as usize, h as usize);

    let ctx = egui::Context::default();
    let mut worker = HealWorker::spawn(&ctx);
    let base = Arc::new(base);

    // UI-thread cost: this is all the stroke-release handler does now.
    let t = Instant::now();
    worker.enqueue(HealRequest::Rebuild {
        revision: 1,
        base: Arc::clone(&base),
        spots: vec![stroke],
    });
    let enqueue_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Worker latency: time from enqueue to the healed reply.
    let t = Instant::now();
    let reply = worker.rx.recv_timeout(Duration::from_secs(60));
    let worker_ms = t.elapsed().as_secs_f64() * 1000.0;
    let ok = matches!(reply, Ok(HealReply::Healed { .. }));

    println!("\n=== heal async offload {w}x{h} (stroke size 80) ===");
    println!("UI enqueue (blocks release) {:>8.3} ms", enqueue_ms);
    println!("worker rebuild latency      {:>8.1} ms", worker_ms);
    println!("reply healed                {:>8}", ok);
}

/// The per-frame costs removed by upload caching — building the original
/// (before) texture `ColorImage` and rebuilding the output histogram — measured at
/// the preview size. Both are now skipped when the base/pixels are unchanged.
/// `cargo test --release -- --ignored bench_upload_redundancy --nocapture`
#[test]
#[ignore]
fn bench_upload_redundancy() {
    let (w, h) = (1200usize, 800usize);
    let buf = synth_image(w, h);
    let reps = 50;

    let t = Instant::now();
    for _ in 0..reps {
        let img = egui::ColorImage::from_rgb([w, h], &buf);
        std::hint::black_box(&img);
    }
    let build_ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    let t = Instant::now();
    for _ in 0..reps {
        let hist = super::histogram::Histogram::of_rgb8(&buf);
        std::hint::black_box(&hist);
    }
    let hist_ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    println!("\n=== upload redundancy {w}x{h} (avg of {reps}) ===");
    println!("ColorImage::from_rgb (orig)  {:>7.3} ms", build_ms);
    println!("Histogram::of_rgb8           {:>7.3} ms", hist_ms);
}

/// The "baked LUT size" trade-off: how much the image differs at
/// different sizes (relative to 64³) and what baking costs. The external LUT is
/// 33³ (a typical .cube); settings are strong (all sliders) and moderate.
/// `cargo test --release -- --ignored lut_size_tradeoff --nocapture`
#[test]
#[ignore]
fn lut_size_tradeoff() {
    let external = synth_lut(33);

    // --- Moderate edits: closer to typical editing. ---
    let mut mild = FilterSettings::default();
    mild.exposure = 0.2;
    mild.contrast = 10.0;
    mild.highlights = -10.0;
    mild.shadows = 12.0;
    mild.temp = 8.0;
    mild.vibrance = 15.0;
    mild.saturation = 5.0;
    mild.texture = 15.0;
    mild.clarity = 12.0;
    mild.sharpen = 20.0;
    mild.lut_intensity = 50.0;

    let (w, h) = (512usize, 512usize);
    let gradient = smooth_gradient(w, h);

    for (label, settings) in [("strong", baseline_settings()), ("mild", mild)] {
        println!("\n=== LUT: {label} (sliders + external LUT 33³) ===");
        println!(
            "{:>5} {:>9} {:>8} {:>9} {:>8}",
            "size", "bake ms", "max|Δ|", "mean|Δ|", "≥2, %"
        );

        // The reference is a DIRECT per-pixel pass (the function the LUT
        // approximates). Each size is compared against it.
        let mut truth = gradient.clone();
        TinyLumaApp::run_color_pass(&settings, Some(&external), &mut truth);

        for size in [33usize, 48, 64] {
            let t = Instant::now();
            let lut = TinyLumaApp::bake_lut(&settings, Some(&external), size);
            let bake = t.elapsed();
            let mut out = gradient.clone();
            apply_lut(&lut, &mut out);

            let mut max_diff = 0i32;
            let mut sum = 0f64;
            let mut ge2 = 0usize;
            let mut max_at = [0u8; 3];
            let mut max_a = [0u8; 3];
            let mut max_b = [0u8; 3];
            for (i, (a, b)) in out.iter().zip(truth.iter()).enumerate() {
                let d = (*a as i32 - *b as i32).abs();
                sum += d as f64;
                if d >= 2 {
                    ge2 += 1;
                }
                if d > max_diff {
                    max_diff = d;
                    let px = i / 3;
                    for c in 0..3 {
                        max_at[c] = gradient[px * 3 + c];
                        max_a[c] = out[px * 3 + c];
                        max_b[c] = truth[px * 3 + c];
                    }
                }
            }
            let n = out.len() as f64;
            println!(
                "{:>5} {:>9.1} {:>8} {:>9.3} {:>8.3}",
                format!("{size}³"),
                bake.as_secs_f64() * 1000.0,
                max_diff,
                sum / n,
                ge2 as f64 / n * 100.0,
            );
            if size != 64 {
                println!(
                    "        max @ in {:?}: {}³ -> {:?}, truth -> {:?}",
                    max_at, size, max_a, max_b
                );
            }
        }
    }
}

/// The reduced-resolution interactive drag tier. A full-resolution
    /// spatial drag frame vs the drag tier (staged detail + effects on a
    /// downscaled input) and the effects-only variant (detail head reused, what an
    /// effects-slider drag pays), at the 1200px preview cap.
    /// `cargo test --release -- --ignored bench_drag_tier --nocapture`
    #[test]
    #[ignore]
    fn bench_drag_tier() {
        use std::time::Instant;

        let (w, h) = (1200usize, 800usize);
        let s = baseline_settings();
        let color = synth_image(w, h);
        let luma = TinyLumaApp::compute_luma_map(&color);
        let (clarity, csw, csh) = TinyLumaApp::compute_clarity_cache(&luma, w, h);
        let grain = TinyLumaApp::generate_grain_map(w, h, 42, s.grain_size, s.grain_roughness > 0.0);
        let base = TinyLumaApp::compute_spatial_base(&s, &color, &luma, w, h);

        let mut out = vec![0u8; color.len()];
        let t = Instant::now();
        TinyLumaApp::run_spatial_pass_with_base(
            &s, &color, &mut out, &grain, &clarity, csw, csh, &base, w, h,
        );
        let full_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Drag tier at the reduced resolution (bases cached — a warm drag frame).
        let (dw, dh) = TinyLumaApp::drag_dims(w, h);
        let small = TinyLumaApp::downscale_rgb(&color, w, h, dw, dh);
        let dluma = TinyLumaApp::compute_luma_map(&small);
        let (dclarity, ds, dsh) = TinyLumaApp::compute_clarity_cache(&dluma, dw, dh);
        let dbase = TinyLumaApp::compute_spatial_base(&s, &small, &dluma, dw, dh);
        let dgrain =
            TinyLumaApp::generate_grain_map(dw, dh, 42, s.grain_size, s.grain_roughness > 0.0);
        let mut detail = vec![0.0f32; dw * dh * 3];
        let mut dout = vec![0u8; dw * dh * 3];

        let t = Instant::now();
        TinyLumaApp::spatial_detail_pass(&s, &small, &dclarity, ds, dsh, &dbase, dw, dh, &mut detail);
        TinyLumaApp::spatial_effects_pass(&s, &detail, &dgrain, &dbase, dw, dh, &mut dout);
        let tier_detail_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Effects-only frame: the detail head is reused (what an effects-slider
        // drag pays — e.g. Glow/Halation/Flare/Grain/Vignette).
        let t = Instant::now();
        TinyLumaApp::spatial_effects_pass(&s, &detail, &dgrain, &dbase, dw, dh, &mut dout);
        let tier_effects_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Grain-map regeneration: the dominant cost of a Grain Size drag.
        let reps = 5;
        let t = Instant::now();
        for _ in 0..reps {
            let g = TinyLumaApp::generate_grain_map(w, h, 42, s.grain_size, s.grain_roughness > 0.0);
            std::hint::black_box(&g);
        }
        let grain_full_ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        let t = Instant::now();
        for _ in 0..reps {
            let g = TinyLumaApp::generate_grain_map(dw, dh, 42, s.grain_size, s.grain_roughness > 0.0);
            std::hint::black_box(&g);
        }
        let grain_drag_ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;

        println!("\n=== drag tier {w}x{h} -> {dw}x{dh} ===");
        println!("full-res spatial drag    {:>8.2} ms", full_ms);
        println!("drag tier (detail+fx)    {:>8.2} ms", tier_detail_ms);
        println!("drag tier (effects only) {:>8.2} ms", tier_effects_ms);
        println!("grain regen full-res     {:>8.2} ms", grain_full_ms);
        println!("grain regen drag-res     {:>8.2} ms", grain_drag_ms);
    }
