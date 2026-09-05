use crate::drm::{DrmSyncobj, WAIT_RELEASE_TIMEOUT_NS};
use crate::Result;

/// The Swapchain is responsible only for the double-buffered DMA-BUF slots: it
/// tracks which slot is current and records each slot's release point so it can
/// apply backpressure (throttling via release-point waits) before reusing a
/// slot.
///
/// It does not own the DRM syncobj timeline nor the Vulkan renderer — those
/// are owned by the caller (main), which passes the [`DrmSyncobj`] in so the
/// swapchain can wait on buffer release. Drawing and sync-file bridging are the
/// caller's responsibility.
pub struct Swapchain {
    /// Release point for each buffer slot, set after the first presentation.
    release_points: [Option<u64>; 2],
    /// Which buffer slot we are currently using (0 or 1).
    current_idx: usize,
    /// Whether each slot has been presented at least once.
    presented: [bool; 2],
}

impl Swapchain {
    pub fn new() -> Self {
        Self {
            release_points: [None, None],
            current_idx: 0,
            presented: [false, false],
        }
    }

    /// Acquire the next buffer slot. If the slot was previously presented it
    /// waits for the compositor to release it (via the caller-owned
    /// `drm_sync` timeline) before returning the slot index.
    pub fn acquire(&mut self, drm_sync: &DrmSyncobj) -> Result<usize> {
        // 1. Rotate buffer slot.
        self.current_idx = (self.current_idx + 1) % 2;

        // 2. Backpressure: wait on the release point of the buffer we are
        //    about to reuse. Only a buffer that has been previously presented
        //    carries a release point; the first draw into a slot has nothing
        //    to wait on.
        if self.presented[self.current_idx]
            && let Some(release_point) = self.release_points[self.current_idx]
        {
            drm_sync.wait_available(release_point, WAIT_RELEASE_TIMEOUT_NS)?;
        }

        Ok(self.current_idx)
    }

    /// Record the release point for a slot once it has been presented, so the
    /// next `acquire` of that slot can wait for it.
    pub fn mark_presented(&mut self, slot: usize, release_point: u64) {
        self.release_points[slot] = Some(release_point);
        self.presented[slot] = true;
    }
}

impl Default for Swapchain {
    fn default() -> Self {
        Self::new()
    }
}
