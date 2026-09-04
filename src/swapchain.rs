use std::os::unix::io::OwnedFd;

use ash::vk;

use crate::drm::{DrmSyncobj, WAIT_RELEASE_TIMEOUT_NS, open_render_node};
use crate::renderer::vulkan::Vulkan;
use crate::{DmaFrame, ExplicitSync};

/// The Swapchain coordinates between the Vulkan Renderer and the DRM syncobj
/// Timeline. It manages double-buffered DMA-BUF slots, handles backpressure
/// (throttling via release-point waits), and bridges the Vulkan sync-file to
/// the DRM syncobj timeline.
///
/// The Swapchain does not own the Renderer — it receives it by mutable
/// reference, keeping the modules decoupled.
pub struct Swapchain {
    drm_sync: DrmSyncobj,
    /// Release point for each buffer slot, set after the first presentation.
    release_points: [Option<u64>; 2],
    /// Which buffer slot we are currently using (0 or 1).
    current_idx: usize,
    /// Whether each slot has been presented at least once.
    presented: [bool; 2],
}

impl Swapchain {
    pub fn new(main_device: u64) -> std::io::Result<Self> {
        let fd = open_render_node(main_device)?;
        let drm_sync = DrmSyncobj::create(fd)?;
        Ok(Self {
            drm_sync,
            release_points: [None, None],
            current_idx: 0,
            presented: [false, false],
        })
    }

    /// Export the DRM syncobj timeline fd so the compositor can import it via
    /// `wp_linux_drm_syncobj_manager_v1.import_timeline`.
    pub fn timeline_fd(&self) -> std::io::Result<OwnedFd> {
        self.drm_sync.export_fd()
    }

    /// Produce the next frame: wait on backpressure, ask the Renderer to draw,
    /// and bridge the resulting sync-file to the DRM syncobj timeline.
    ///
    /// The caller provides a `record` closure that receives the device,
    /// command buffer, and image handle to record GPU commands
    /// (see [`Vulkan::draw`]).
    ///
    /// Returns the DMA-BUF frame metadata and the explicit synchronization
    /// points for the Wayland backend to apply at commit time.
    pub fn next_frame(
        &mut self,
        renderer: &mut Vulkan,
        width: u32,
        height: u32,
        record: impl FnOnce(usize, ash::Device, vk::CommandBuffer, vk::Image),
    ) -> Result<(DmaFrame, ExplicitSync), String> {
        // 1. Rotate buffer slot.
        self.current_idx = (self.current_idx + 1) % 2;

        // 2. Backpressure: wait on the release point of the buffer we are
        //    about to reuse. Only a buffer that has been previously presented
        //    carries a release point; the first draw into a slot has nothing
        //    to wait on.
        if self.presented[self.current_idx]
            && let Some(release_point) = self.release_points[self.current_idx]
        {
            self.drm_sync
                .wait_available(release_point, WAIT_RELEASE_TIMEOUT_NS)
                .map_err(|e| format!("buffer reuse wait: {e}"))?;
        }

        // 3. Production: ask the Renderer to draw into the current buffer.
        let (frame, sync_file) = renderer.draw(self.current_idx, width, height, record)?;

        // 4. Sync logic: import the Vulkan completion fence (sync-file) into
        //    the DRM syncobj timeline. The compositor waits on the returned
        //    acquire point before sampling the buffer, and signals the
        //    following release point when it is done with it.
        let acquire_point = self
            .drm_sync
            .import_sync_file(&sync_file)
            .map_err(|e| format!("import sync-file: {e}"))?;
        let release_point = acquire_point + 1;

        // 5. Track release point for the next reuse of this buffer slot.
        self.release_points[self.current_idx] = Some(release_point);
        self.presented[self.current_idx] = true;

        // 6. Export the timeline fd for the Wayland backend.
        let timeline_fd = self
            .drm_sync
            .export_fd()
            .map_err(|e| format!("export timeline fd: {e}"))?;

        Ok((
            frame,
            ExplicitSync {
                timeline_fd,
                acquire_point,
                release_point,
            },
        ))
    }
}
