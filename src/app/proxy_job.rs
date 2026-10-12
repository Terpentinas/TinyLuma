// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! Background viewport-proxy render worker.
//!
//! Mirrors [`super::heal_job`]: one long-lived worker thread, an `Arc` control
//! block with atomics, and a bounded channel so **at most one** job is in flight.
//! The debounced proxy rebuild ([`super::processing`]'s `maybe_rebuild_proxy`)
//! used to downscale *and* re-render the whole preview on the UI thread after the
//! `PROXY_SETTLE` debounce — a one-off stall that grows with area (~204 ms at the
//! 2560 bucket). Now the UI only enqueues; the worker downscales the higher-res
//! source, replays the retouch layer, runs the pure preview render
//! ([`super::TinyLumaApp::render_preview_at`], the same 33³ preview path the UI
//! uses) and returns the processed pixels plus the new raw working proxy.
//!
//! Coalescing and cancellation are by `revision`, exactly as in the heal worker: a
//! superseded job is dropped before any work, and a stale reply is discarded by the
//! UI. A newer job subsumes an older one because the reply carries the full proxy.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};

use eframe::egui;
use image::RgbImage;

use super::TinyLumaApp;
use crate::lut::Lut3D;
use crate::retouch::RetouchLayer;
use crate::settings::FilterSettings;

/// Control block shared with the worker (same shape as `HealControl`).
pub(crate) struct ProxyControl {
    /// Latest requested revision; the worker drops anything older before work.
    revision: AtomicU64,
    /// Set on drop / cancel: the worker stops and discards.
    cancel: AtomicBool,
}

impl ProxyControl {
    fn new() -> Self {
        Self {
            revision: AtomicU64::new(0),
            cancel: AtomicBool::new(false),
        }
    }
}

/// One unit of work for the proxy worker: derive the working proxy from the
/// higher-res source, heal it with the retouch layer, then render the preview.
pub(crate) struct ProxyRequest {
    pub(crate) revision: u64,
    /// Higher-res decoded source (long side ≤ `SOURCE_WORK_MAX`).
    pub(crate) source: Arc<RgbImage>,
    /// Target long-side bucket dimensions (already clamped to the source).
    pub(crate) target: (usize, usize),
    /// Filter settings snapshot at enqueue time.
    pub(crate) settings: FilterSettings,
    /// External LUT snapshot (`Arc` share, so enqueuing is a pointer clone). The
    /// worker bakes its own preview grid from it.
    pub(crate) lut: Option<Arc<Lut3D>>,
    /// Retouch layer snapshot (normalized coordinates, resolution-independent).
    pub(crate) layer: Arc<RetouchLayer>,
}

impl ProxyRequest {
    fn revision(&self) -> u64 {
        self.revision
    }
}

/// The worker's reply: the new raw working proxy (installed as `preview_base`,
/// un-healed so the UI's `healed_base` stays the single place that applies the
/// layer) plus the processed preview pixels for that resolution.
pub(crate) enum ProxyReply {
    Rendered {
        revision: u64,
        proxy: RgbImage,
        processed: Vec<u8>,
        dims: (usize, usize),
    },
}

/// UI-side handle to the worker: the send/recv ends, the shared control, and at
/// most one coalesced pending job that could not be sent because the worker was
/// still busy.
pub(crate) struct ProxyWorker {
    tx: mpsc::SyncSender<ProxyRequest>,
    pub(crate) rx: mpsc::Receiver<ProxyReply>,
    control: Arc<ProxyControl>,
    /// Newest job superseded while the channel was full; sent by `pump`.
    pending: Option<ProxyRequest>,
}

impl ProxyWorker {
    /// Spawns the single worker thread. `ctx` is used only to wake the UI when a
    /// reply is ready (the worker must not block on the UI being idle).
    pub(crate) fn spawn(ctx: &egui::Context) -> Self {
        // Bound of 1: at most one job queued, so the source/proxy buffers cannot
        // pile up. A newer job replaces the pending one.
        let (tx, req_rx) = mpsc::sync_channel::<ProxyRequest>(1);
        let (rep_tx, rx) = mpsc::channel::<ProxyReply>();
        let control = Arc::new(ProxyControl::new());
        let ctl = control.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || worker_loop(ctl, req_rx, rep_tx, ctx));
        Self {
            tx,
            rx,
            control,
            pending: None,
        }
    }

    /// Enqueues a job as the newest revision. If the worker is still busy the job
    /// is kept (replacing any older pending job) and retried by `pump`.
    pub(crate) fn enqueue(&mut self, req: ProxyRequest) {
        self.control.revision.store(req.revision(), Ordering::Relaxed);
        match self.tx.try_send(req) {
            Ok(()) | Err(mpsc::TrySendError::Disconnected(_)) => {}
            Err(mpsc::TrySendError::Full(req)) => self.pending = Some(req),
        }
    }

    /// Retries a coalesced pending job. Called once per UI frame.
    pub(crate) fn pump(&mut self) {
        if let Some(req) = self.pending.take() {
            match self.tx.try_send(req) {
                Ok(()) | Err(mpsc::TrySendError::Disconnected(_)) => {}
                Err(mpsc::TrySendError::Full(req)) => self.pending = Some(req),
            }
        }
    }

    /// Bumps the revision so in-flight replies are treated as stale, and drops the
    /// coalesced job. Does not join the thread (a running render finishes).
    pub(crate) fn invalidate(&mut self) {
        self.control.revision.fetch_add(1, Ordering::Relaxed);
        self.pending = None;
    }
}

impl Drop for ProxyWorker {
    fn drop(&mut self) {
        self.control.cancel.store(true, Ordering::Relaxed);
    }
}

/// The worker loop.
fn worker_loop(
    control: Arc<ProxyControl>,
    req_rx: mpsc::Receiver<ProxyRequest>,
    rep_tx: mpsc::Sender<ProxyReply>,
    ctx: egui::Context,
) {
    while let Ok(req) = req_rx.recv() {
        if control.cancel.load(Ordering::Relaxed) {
            break;
        }
        let revision = req.revision();
        // Superseded before we started: skip the expensive work entirely.
        if control.revision.load(Ordering::Relaxed) != revision {
            continue;
        }

        // Downscale the higher-res source to the target bucket (never upscale).
        let (sw, sh) = (req.source.width() as usize, req.source.height() as usize);
        let (mut dw, mut dh) = req.target;
        if dw > sw || dh > sh {
            dw = sw;
            dh = sh;
        }
        dw = dw.max(1);
        dh = dh.max(1);
        let raw = TinyLumaApp::downscale_rgb(req.source.as_raw(), sw, sh, dw, dh);
        let Some(proxy) = RgbImage::from_raw(dw as u32, dh as u32, raw) else {
            continue;
        };

        // Heal a copy for the render; the raw proxy is what becomes `preview_base`.
        let healed = if req.layer.is_empty() {
            proxy.clone()
        } else {
            let mut m = proxy.clone();
            req.layer.apply_to(&mut m);
            m
        };
        let processed = TinyLumaApp::render_preview_at(&healed, &req.settings, req.lut.as_deref());

        let reply = ProxyReply::Rendered {
            revision,
            proxy,
            processed,
            dims: (dw, dh),
        };
        if rep_tx.send(reply).is_err() {
            break;
        }
        ctx.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            let v = ((x * 7 + y * 3) % 256) as u8;
            image::Rgb([v, 255 - v, (v / 2).wrapping_add(30)])
        })
    }

    /// The worker's reply must equal the pure preview render on the same base,
    /// and its raw proxy must equal a direct downscale — across ladder buckets.
    /// This is the parity discipline from `spatial_stages_match_monolithic`: the
    /// background proxy render and the UI render share one implementation, so the
    /// zoom-settle swap cannot change the pixels.
    #[test]
    fn worker_render_matches_pure_preview_render() {
        let ctx = egui::Context::default();
        let (sw, sh) = (1600usize, 1200usize);
        let src = Arc::new(synth(sw as u32, sh as u32));

        let settings = FilterSettings {
            exposure: 25.0,
            contrast: 20.0,
            texture: 40.0,
            clarity: 30.0,
            sharpen: 25.0,
            ..FilterSettings::default()
        };

        let mut worker = ProxyWorker::spawn(&ctx);
        for (i, &bucket) in [640usize, 960, 1280, 1600].iter().enumerate() {
            let (dw, dh) = TinyLumaApp::proxy_dims_for_long(sw, sh, bucket);
            let revision = (i as u64) + 1;
            worker.enqueue(ProxyRequest {
                revision,
                source: src.clone(),
                target: (dw, dh),
                settings,
                lut: None,
                layer: Arc::new(RetouchLayer::default()),
            });
            let reply = worker
                .rx
                .recv_timeout(std::time::Duration::from_secs(60))
                .expect("proxy worker reply");
            let ProxyReply::Rendered {
                revision: got,
                proxy,
                processed,
                dims,
            } = reply;
            assert_eq!(got, revision);
            assert_eq!(dims, (dw, dh));

            // The raw proxy equals a direct downscale of the source.
            let raw = TinyLumaApp::downscale_rgb(src.as_raw(), sw, sh, dw, dh);
            assert_eq!(proxy.as_raw(), &raw, "bucket {bucket}: raw proxy");

            // The worker's processed pixels equal the pure preview render.
            let expected = TinyLumaApp::render_preview_at(&proxy, &settings, None);
            assert_eq!(processed, expected, "bucket {bucket}: preview parity");
        }
    }
}
