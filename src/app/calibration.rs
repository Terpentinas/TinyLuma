// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! Numeric calibration kit (test-only).
//!
//! Compares TinyLuma's per-pixel operators against ported reference
//! implementations from the donor projects over a dense colour chart, and
//! reports error metrics. Two kinds of test live here:
//!
//! - **locking tests** (not ignored) assert the current constants stay within
//!   tolerance, so a regression fails `cargo test`;
//! - the **report** (`#[ignore]`) prints error tables, fits the scalar
//!   constants against the reference, and writes CSVs to `target/calibration/`.
//!   Run it with:
//!
//!   ```text
//!   cargo test calibration -- --nocapture --ignored
//!   ```
//!
//! References are ported mathematically, not copied from source. Colour grading
//! is compared against **lightcraft** (`ColorOps::grade`, the Oklab model our
//! implementation follows); brightness against **RapidRAW**'s filmic luma
//! curve; hue is checked to be an exact Oklab rotation.

use std::f32::consts::TAU;
use std::fmt::Write as _;

use crate::color::oklab::{rgb_to_oklab, srgb_to_linear};
use crate::pipeline::grading::{
    ColorGrading, GradingParams, GradingWheel, apply_grading, hsv_to_rgb,
};
use crate::pipeline::light::{BRIGHTNESS_STRENGTH_NEG, BRIGHTNESS_STRENGTH_POS, brightness_curve};

// ─────────────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean distance in Oklab (a ΔE proxy; the space is already perceptual).
fn delta_e(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

struct Stats {
    mean: f32,
    max: f32,
    p95: f32,
}

fn stats(mut v: Vec<f32>) -> Stats {
    if v.is_empty() {
        return Stats {
            mean: 0.0,
            max: 0.0,
            p95: 0.0,
        };
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let mean = v.iter().sum::<f32>() / n as f32;
    let max = *v.last().unwrap();
    let p95 = v[((n as f32 * 0.95) as usize).min(n - 1)];
    Stats { mean, max, p95 }
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour chart
// ─────────────────────────────────────────────────────────────────────────────

/// Dense Oklab chart: lightness × hue × chroma, including near-neutral colours.
fn oklab_chart() -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    for &l in &[0.08f32, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95] {
        for h in 0..32 {
            let h = h as f32 / 32.0 * TAU;
            for &c in &[0.01f32, 0.05, 0.12, 0.22] {
                out.push([l, c * h.cos(), c * h.sin()]);
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference: lightcraft colour grading (`crates/pipeline/src/colorops.rs`)
// ─────────────────────────────────────────────────────────────────────────────

const LIGHTCRAFT_SAT_SCALE: f32 = 0.09;
const LIGHTCRAFT_LUM_SCALE: f32 = 0.12;

fn srgb_hue_to_oklab_hue(deg: f32) -> f32 {
    let [r, g, b] = hsv_to_rgb(deg, 1.0, 1.0);
    let lab = rgb_to_oklab(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b));
    lab[2].atan2(lab[1])
}

fn lightcraft_wheel(w: &GradingWheel) -> [f32; 3] {
    let s = (w.sat / 100.0) * LIGHTCRAFT_SAT_SCALE;
    let h = srgb_hue_to_oklab_hue(w.hue);
    [s * h.cos(), s * h.sin(), (w.lum / 100.0) * LIGHTCRAFT_LUM_SCALE]
}

fn smooth(x0: f32, x1: f32, x: f32) -> f32 {
    let t = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Port of lightcraft's `ColorOps::grade` on an Oklab pixel.
fn lightcraft_grade(lab: [f32; 3], g: &ColorGrading) -> [f32; 3] {
    let wheels = [
        lightcraft_wheel(&g.shadows),
        lightcraft_wheel(&g.midtones),
        lightcraft_wheel(&g.highlights),
        lightcraft_wheel(&g.global),
    ];
    let blending = (g.blending / 100.0).clamp(0.0, 1.0);
    let balance = (g.balance / 100.0).clamp(-1.0, 1.0);
    let m = 0.5 - balance * 0.25;
    let width = 0.15 + blending * 0.5;
    let mut lab = lab;
    let ws = 1.0 - smooth(m - width, m + width * 0.25, lab[0]);
    let wh = smooth(m - width * 0.25, m + width, lab[0]);
    let wm = (1.0 - ws - wh).max(0.0);
    for (k, wt) in wheels.iter().zip([ws, wm, wh, 1.0]) {
        lab[1] += k[0] * wt;
        lab[2] += k[1] * wt;
        lab[0] += k[2] * wt;
    }
    lab
}

fn grading_samples() -> Vec<ColorGrading> {
    vec![
        ColorGrading {
            shadows: GradingWheel {
                hue: 220.0,
                sat: 70.0,
                lum: 0.0,
            },
            ..Default::default()
        },
        ColorGrading {
            highlights: GradingWheel {
                hue: 40.0,
                sat: 60.0,
                lum: 0.0,
            },
            ..Default::default()
        },
        ColorGrading {
            global: GradingWheel {
                hue: 300.0,
                sat: 80.0,
                lum: 20.0,
            },
            ..Default::default()
        },
        ColorGrading {
            shadows: GradingWheel {
                hue: 200.0,
                sat: 50.0,
                lum: -20.0,
            },
            midtones: GradingWheel {
                hue: 120.0,
                sat: 40.0,
                lum: 10.0,
            },
            highlights: GradingWheel {
                hue: 30.0,
                sat: 60.0,
                lum: 20.0,
            },
            blending: 30.0,
            balance: 25.0,
            ..Default::default()
        },
    ]
}

/// Mean ΔE statistics between our grading (with the given scales) and lightcraft.
fn grading_stats(sat_scale: f32, lum_scale: f32, chart: &[[f32; 3]]) -> Stats {
    let mut errs = Vec::new();
    for g in grading_samples() {
        let params = GradingParams::with_scales(&g, sat_scale, lum_scale);
        for &lab in chart {
            let mut ours = lab;
            apply_grading(&mut ours, &params);
            errs.push(delta_e(ours, lightcraft_grade(lab, &g)));
        }
    }
    stats(errs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference: RapidRAW filmic brightness luma (`shader.wgsl`)
// ─────────────────────────────────────────────────────────────────────────────

/// Port of RapidRAW's `apply_filmic_exposure` luma shaping (`RATIONAL_CURVE_MIX
/// = 1`, so the direct term is 0), applied to a display-referred luma.
fn rapidraw_brightness_luma(l: f32, amount: f32) -> f32 {
    if amount == 0.0 || l <= 0.0 {
        return l;
    }
    const MIDTONE_STRENGTH: f32 = 1.2;
    const TOP_ANCHOR: f32 = 1.0;
    let k = 2.0f32.powf(-amount * MIDTONE_STRENGTH);
    let a = l.abs();
    if a <= TOP_ANCHOR {
        let n = a / TOP_ANCHOR;
        (n / (n + (1.0 - n) * k)) * TOP_ANCHOR
    } else {
        TOP_ANCHOR + (a - TOP_ANCHOR) * k
    }
}

/// TinyLuma's brightness model with separate positive/negative strengths.
fn model_brightness(l: f32, amount: f32, pos: f32, neg: f32) -> f32 {
    if amount.abs() < 1e-6 || l <= 0.0 {
        return l;
    }
    let w = (4.0 * l * (1.0 - l)).clamp(0.0, 1.0);
    let gamma = if amount > 0.0 {
        1.0 / (1.0 + amount * w * pos)
    } else {
        1.0 + (-amount) * w * neg
    };
    l.powf(gamma)
}

fn brightness_amounts() -> [f32; 8] {
    [0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0]
}

fn brightness_mean_error(pos: f32, neg: f32, lgrid: &[f32]) -> f32 {
    let mut errs = Vec::new();
    for amount in brightness_amounts() {
        for &l in lgrid {
            let m = model_brightness(l, amount, pos, neg);
            errs.push((m - rapidraw_brightness_luma(l, amount)).abs());
        }
    }
    stats(errs).mean
}

/// Mean |ΔL| over one sign's amounts only (so the two signs fit independently).
fn brightness_signed_error(pos: f32, neg: f32, lgrid: &[f32], positive: bool) -> f32 {
    let mut errs = Vec::new();
    for amount in brightness_amounts() {
        if (amount > 0.0) != positive {
            continue;
        }
        for &l in lgrid {
            let m = model_brightness(l, amount, pos, neg);
            errs.push((m - rapidraw_brightness_luma(l, amount)).abs());
        }
    }
    stats(errs).mean
}

fn fit_brightness(lgrid: &[f32], positive: bool) -> (f32, f32) {
    // Grid search over one sign's strength; the other stays at the production
    // value. Errors are measured on that sign alone.
    let held = if positive {
        BRIGHTNESS_STRENGTH_NEG
    } else {
        BRIGHTNESS_STRENGTH_POS
    };
    let mut best = (held, f32::INFINITY);
    let mut cand = 0.30f32;
    while cand <= 2.0 {
        let (pos, neg) = if positive { (cand, held) } else { (held, cand) };
        let e = brightness_signed_error(pos, neg, lgrid, positive);
        if e < best.1 {
            best = (cand, e);
        }
        cand += 0.01;
    }
    best
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference: RapidRAW hue shift (`shader.wgsl`) — informational
// ─────────────────────────────────────────────────────────────────────────────

/// RapidRAW shifts hue in HSV after converting to sRGB. Returns the Oklab hue
/// (radians) after a shift, for comparison with our Oklab rotation.
fn rapidraw_hue_shift_oklab_hue(oklab: [f32; 3], shift_deg: f32) -> f32 {
    let lin = crate::color::oklab::oklab_to_linear_rgb(oklab[0], oklab[1], oklab[2]);
    let srgb = lin.map(crate::color::oklab::linear_to_srgb);
    // rgb -> hsv
    let max = srgb[0].max(srgb[1]).max(srgb[2]);
    let min = srgb[0].min(srgb[1]).min(srgb[2]);
    let d = max - min;
    let hue = if d < 1e-6 {
        0.0
    } else if max == srgb[0] {
        60.0 * (((srgb[1] - srgb[2]) / d) % 6.0)
    } else if max == srgb[1] {
        60.0 * ((srgb[2] - srgb[0]) / d + 2.0)
    } else {
        60.0 * ((srgb[0] - srgb[1]) / d + 4.0)
    }
    .rem_euclid(360.0);
    let shifted = (hue + shift_deg).rem_euclid(360.0);
    // hsv -> rgb at the original s/v, then back to Oklab.
    let c = d;
    let x = c * (1.0 - ((shifted / 60.0) % 2.0 - 1.0).abs());
    let m = min;
    let (r, g, b) = match (shifted / 60.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let rgb = [r + m, g + m, b + m].map(srgb_to_linear);
    let lab = rgb_to_oklab(rgb[0], rgb[1], rgb[2]);
    let mut h = lab[2].atan2(lab[1]);
    if h < 0.0 {
        h += TAU;
    }
    h
}

// ─────────────────────────────────────────────────────────────────────────────
// Locking tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn grading_matches_lightcraft_within_tolerance() {
    let chart = oklab_chart();
    let err = grading_stats(
        crate::pipeline::grading::SAT_SCALE,
        crate::pipeline::grading::LUM_SCALE,
        &chart,
    );
    assert!(
        err.mean < 0.01 && err.max < 0.02,
        "grading drift vs lightcraft: mean ΔE = {:.4}, max = {:.4}",
        err.mean,
        err.max
    );
}

#[test]
fn brightness_matches_rapidraw_within_tolerance() {
    let lgrid: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
    let err = brightness_mean_error(BRIGHTNESS_STRENGTH_POS, BRIGHTNESS_STRENGTH_NEG, &lgrid);
    assert!(
        err < 0.02,
        "brightness drift vs RapidRAW: mean |ΔL| = {err:.4}"
    );
}

#[test]
fn brightness_curve_matches_production() {
    // `apply_brightness` must go through `brightness_curve` unchanged.
    for amount in [-1.0f32, -0.5, 0.5, 1.0] {
        for &l in &[0.1f32, 0.3, 0.5, 0.7, 0.9] {
            let mut p = [l, 0.0, 0.0];
            crate::pipeline::light::apply_brightness(&mut p, amount);
            let strength = if amount > 0.0 {
                BRIGHTNESS_STRENGTH_POS
            } else {
                BRIGHTNESS_STRENGTH_NEG
            };
            assert!(
                (p[0] - brightness_curve(l, amount, strength)).abs() < 1e-6,
                "amount {amount} l {l}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Report
// ─────────────────────────────────────────────────────────────────────────────

/// Full numeric report + constant fits. Writes CSVs to `target/calibration/`.
#[test]
#[ignore]
fn calibration_report() {
    let chart = oklab_chart();
    let mut out = String::new();

    // --- Colour grading ---
    let _ = writeln!(out, "\n=== Colour grading vs lightcraft ({} samples) ===", chart.len());
    let _ = writeln!(out, "{:>10} {:>10} {:>10} {:>10}", "sat", "lum", "meanΔE", "p95ΔE");
    let mut best_sat = (f32::NAN, f32::INFINITY);
    let mut best_lum = (f32::NAN, f32::INFINITY);
    for i in 0..=40 {
        let sat = 0.05 + i as f32 * 0.005;
        let e = grading_stats(sat, crate::pipeline::grading::LUM_SCALE, &chart).mean;
        if e < best_sat.1 {
            best_sat = (sat, e);
        }
    }
    for i in 0..=40 {
        let lum = 0.06 + i as f32 * 0.005;
        let e = grading_stats(crate::pipeline::grading::SAT_SCALE, lum, &chart).mean;
        if e < best_lum.1 {
            best_lum = (lum, e);
        }
    }
    let cur = (
        crate::pipeline::grading::SAT_SCALE,
        crate::pipeline::grading::LUM_SCALE,
    );
    let cur_stats = grading_stats(cur.0, cur.1, &chart);
    let _ = writeln!(
        out,
        "{:>10.3} {:>10.3} {:>10.5} {:>10.5}   <- current",
        cur.0, cur.1, cur_stats.mean, cur_stats.p95
    );
    let _ = writeln!(
        out,
        "best sat = {:.3} (meanΔE {:.5}); best lum = {:.3} (meanΔE {:.5})",
        best_sat.0, best_sat.1, best_lum.0, best_lum.1
    );

    // --- Brightness ---
    let lgrid: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
    let cur_b = brightness_mean_error(BRIGHTNESS_STRENGTH_POS, BRIGHTNESS_STRENGTH_NEG, &lgrid);
    let (best_pos, err_pos) = fit_brightness(&lgrid, true);
    let (best_neg, err_neg) = fit_brightness(&lgrid, false);
    let _ = writeln!(out, "\n=== Brightness vs RapidRAW filmic luma (|ΔL|) ===");
    let _ = writeln!(
        out,
        "current strength pos {} / neg {}: mean |ΔL| = {:.5}",
        BRIGHTNESS_STRENGTH_POS, BRIGHTNESS_STRENGTH_NEG, cur_b
    );
    let _ = writeln!(
        out,
        "best positive strength = {:.2} (mean {:.5}); best negative strength = {:.2} (mean {:.5})",
        best_pos, err_pos, best_neg, err_neg
    );

    // --- Hue (informational) ---
    let _ = writeln!(out, "\n=== Hue: exactness & RapidRAW HSV comparison (deg) ===");
    let _ = writeln!(out, "{:>8} {:>16} {:>18}", "shift", "our |angle err|", "vs HSV mean err");
    for shift in [15.0f32, 30.0, 60.0, 90.0, 180.0] {
        let mut own = Vec::new();
        let mut vs = Vec::new();
        let rad = shift.to_radians();
        for lab in oklab_chart() {
            let h0 = lab[2].atan2(lab[1]);
            let mut p = lab;
            crate::pipeline::color::apply_hue_rotation(&mut p, rad);
            let h1 = p[2].atan2(p[1]);
            let mut want = h0 + rad;
            if want < 0.0 {
                want += TAU;
            }
            if want >= TAU {
                want -= TAU;
            }
            let mut got = h1;
            if got < 0.0 {
                got += TAU;
            }
            let mut e = (got - want).abs();
            if e > std::f32::consts::PI {
                e = TAU - e;
            }
            own.push(e.to_degrees());
            let hsv = rapidraw_hue_shift_oklab_hue(lab, shift);
            let mut d = (hsv - got).abs();
            if d > std::f32::consts::PI {
                d = TAU - d;
            }
            vs.push(d.to_degrees());
        }
        let _ = writeln!(
            out,
            "{:>8.0} {:>16.4} {:>18.3}",
            shift,
            stats(own).mean,
            stats(vs).mean
        );
    }

    // --- Vignette (shape sanity) ---
    let _ = writeln!(out, "\n=== Vignette: mask at normalized points (default) ===");
    let p = crate::pipeline::vignette::VignetteParams::new(
        &crate::pipeline::vignette::VignetteSettings {
            amount: -50.0,
            ..Default::default()
        },
    );
    let w = 1000;
    let h = 700;
    for (label, x, y) in [
        ("centre", w / 2, h / 2),
        ("edge-mid x", w - 1, h / 2),
        ("edge-mid y", w / 2, h - 1),
        ("corner", w - 1, h - 1),
    ] {
        let _ = writeln!(out, "  {:>12}: {:.4}", label, p.mask(x, y, w, h));
    }

    // --- CSV ---
    let dir = std::path::Path::new("target/calibration");
    let _ = std::fs::create_dir_all(dir);
    let mut csv = String::from("sat_scale,mean_de\n");
    for i in 0..=40 {
        let sat = 0.05 + i as f32 * 0.005;
        let _ = writeln!(
            csv,
            "{:.4},{:.6}",
            sat,
            grading_stats(sat, crate::pipeline::grading::LUM_SCALE, &chart).mean
        );
    }
    let _ = std::fs::write(dir.join("grading_sat_scale.csv"), csv);

    let mut csv = String::from("strength,pos_mean,neg_mean\n");
    let mut s = 0.30f32;
    while s <= 2.0 {
        let _ = writeln!(
            csv,
            "{:.3},{:.6},{:.6}",
            s,
            brightness_signed_error(s, BRIGHTNESS_STRENGTH_NEG, &lgrid, true),
            brightness_signed_error(BRIGHTNESS_STRENGTH_POS, s, &lgrid, false)
        );
        s += 0.01;
    }
    let _ = std::fs::write(dir.join("brightness_strength.csv"), csv);
    let _ = writeln!(out, "\nCSVs written to target/calibration/");

    println!("{out}");
}
