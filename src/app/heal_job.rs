// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 ThetaCursed

//! Background full-resolution heal worker.
//!
//! Mirrors the prefetch/export threading pattern
//! ([`super::image_io::PrefetchControl`]): one long-lived worker thread, an
//! `Arc` control block with atomics, and a bounded channel so **at most one**
//! job is in flight. The UI thread never runs the multi-second content-aware
//! fill or the full-resolution decode on its own; it enqueues a job and applies
//! the result on a later frame.
//!
//! Coalescing and cancellation are by `revision`. Every job carries a
//! monotonically increasing revision; the worker drops a job that has already
//! been superseded before doing any expensive work, and the UI discards a reply
//! whose revision is no longer the latest. Because a rebuild always replays the
//! whole layer from the immutable base, a discarded reply is always safe to
//! throw away — the newest job subsumes it.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};

use eframe::egui;
use image::{GenericImageView, RgbImage};

use crate::retouch::raster::Rect;
use crate::retouch::spot::{apply_spot_cached, op_rect, SpotCache};
use crate::retouch::Spot;

/// Control block shared with the worker (same shape as `PrefetchControl`).
pub(crate) struct HealControl {
    /// Latest requested revision; the worker drops anything older before work.
    pub(crate) revision: AtomicU64,
    /// Set on tool close / session change: the worker stops and discards.
    pub(crate) cancel: AtomicBool,
}

impl HealControl {
    fn new() -> Self {
        Self {
            revision: AtomicU64::new(0),
            cancel: AtomicBool::new(false),
        }
    }
}

/// One unit of work for the heal worker.
pub(crate) enum HealRequest {
    /// Decode the full-resolution original (long side ≤ `max_dim`).
    Decode {
        revision: u64,
        path: PathBuf,
        max_dim: u32,
    },
    /// Rebuild the healed full-res image: clone the immutable `base`, then
    /// replay every spot in `spots` through the (worker-owned) content cache.
    Rebuild {
        revision: u64,
        base: Arc<RgbImage>,
        spots: Vec<Spot>,
    },
}

impl HealRequest {
    pub(crate) fn revision(&self) -> u64 {
        match self {
            HealRequest::Decode { revision, .. } | HealRequest::Rebuild { revision, .. } => {
                *revision
            }
        }
    }
}

/// The worker's reply.
pub(crate) enum HealReply {
    /// The decoded full-resolution original, ready to become `full_base`.
    Decoded {
        revision: u64,
        path: PathBuf,
        image: RgbImage,
    },
    /// The rebuilt healed image plus the union of the regions that changed.
    /// `dirty = None` means the layer is empty (image equals the base).
    Healed {
        revision: u64,
        image: RgbImage,
        dirty: Option<Rect>,
    },
    /// Decode failed (unreadable file). The UI clears its loading state.
    Failed { revision: u64 },
}

/// UI-side handle to the worker: the send/recv ends, the shared control, and at
/// most one coalesced pending job that could not be sent because the worker was
/// still busy.
pub(crate) struct HealWorker {
    tx: mpsc::SyncSender<HealRequest>,
    pub(crate) rx: mpsc::Receiver<HealReply>,
    control: Arc<HealControl>,
    /// Newest job superseded while the channel was full; sent by `pump`.
    pending: Option<HealRequest>,
}

impl HealWorker {
    /// Spawns the single worker thread. `ctx` is used only to wake the UI when a
    /// reply is ready (the worker must not block on the UI being idle).
    pub(crate) fn spawn(ctx: &egui::Context) -> Self {
        // Bound of 1: at most one job queued, so a full-res buffer cannot pile up.
        let (tx, req_rx) = mpsc::sync_channel::<HealRequest>(1);
        let (rep_tx, rx) = mpsc::channel::<HealReply>();
        let control = Arc::new(HealControl::new());
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

    /// Enqueues a job as the newest revision. If the worker is still busy the
    /// job is kept (replacing any older pending job) and retried by `pump`.
    pub(crate) fn enqueue(&mut self, req: HealRequest) {
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

    /// Bumps the revision so in-flight replies are treated as stale, and drops
    /// the coalesced job. Does not join the thread (a running fill finishes).
    pub(crate) fn invalidate(&mut self) {
        self.control
            .revision
            .fetch_add(1, Ordering::Relaxed);
        self.pending = None;
    }
}

impl Drop for HealWorker {
    fn drop(&mut self) {
        self.control.cancel.store(true, Ordering::Relaxed);
    }
}

/// The worker loop. Owns the `SpotCache` for its whole lifetime so memoized
/// fills survive coalesced/ superseded jobs.
fn worker_loop(
    control: Arc<HealControl>,
    req_rx: mpsc::Receiver<HealRequest>,
    rep_tx: mpsc::Sender<HealReply>,
    ctx: egui::Context,
) {
    let mut cache = SpotCache::default();
    while let Ok(req) = req_rx.recv() {
        if control.cancel.load(Ordering::Relaxed) {
            break;
        }
        let revision = req.revision();
        // Superseded before we started: skip the expensive work entirely.
        if control.revision.load(Ordering::Relaxed) != revision {
            continue;
        }

        let reply = match req {
            HealRequest::Decode {
                revision,
                path,
                max_dim,
            } => match decode_full(&path, max_dim) {
                Some(image) => HealReply::Decoded {
                    revision,
                    path,
                    image,
                },
                None => HealReply::Failed { revision },
            },
            HealRequest::Rebuild {
                revision,
                base,
                spots,
            } => {
                let mut image = (*base).clone();
                let (w, h) = (image.width() as usize, image.height() as usize);
                let mut dirty: Option<Rect> = None;
                for spot in &spots {
                    apply_spot_cached(&mut cache, &mut image, spot);
                    if let Some(r) = op_rect(w, h, spot) {
                        dirty = Some(match dirty {
                            Some(d) => union(d, r),
                            None => r,
                        });
                    }
                }
                HealReply::Healed {
                    revision,
                    image,
                    dirty,
                }
            }
        };

        if rep_tx.send(reply).is_err() {
            break;
        }
        ctx.request_repaint();
    }
}

/// Decodes `path` and scales its long side to at most `max_dim`.
fn decode_full(path: &std::path::Path, max_dim: u32) -> Option<RgbImage> {
    let img = image::open(path).ok()?;
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return None;
    }
    let scale = (max_dim as f32 / w.max(h) as f32).min(1.0);
    let full = if scale < 1.0 {
        img.resize(
            (w as f32 * scale).round().max(1.0) as u32,
            (h as f32 * scale).round().max(1.0) as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8()
    } else {
        img.to_rgb8()
    };
    Some(full)
}

/// Smallest rectangle containing both `a` and `b`.
fn union(a: Rect, b: Rect) -> Rect {
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    let x1 = (a.x + a.w).max(b.x + b.w);
    let y1 = (a.y + a.h).max(b.y + b.h);
    Rect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    }
}
