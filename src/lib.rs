use std::os::unix::io::OwnedFd;

pub mod backend;
pub mod drm;
pub mod error;
pub mod renderer;
pub mod swapchain;

pub use error::{Result, WayvekError};

/// DRM_FORMAT_MODIFIER_INVALID sentinel used in format negotiation.
pub const DRM_FORMAT_MODIFIER_INVALID: u64 = (1u64 << 56) - 1;

/// Per-plane layout (offset + stride) of an exported DMA-BUF, as reported
/// by vkGetImageSubresourceLayout.
#[derive(Clone, Debug)]
pub struct DmaPlane {
    pub offset: u64,
    pub stride: u64,
}

/// A frame produced by the renderer, expressed entirely in Linux kernel
/// terms: a DMA-BUF file descriptor, dimensions, DRM format/modifier, and
/// per-plane layout. This is the contract between the Renderer and the
/// Wayland backend — no Vulkan or Wayland types cross this boundary.
#[derive(Debug)]
pub struct DmaFrame {
    pub fd: OwnedFd,
    pub width: u32,
    pub height: u32,
    /// DRM Fourcc code (e.g. DRM_FORMAT_XRGB8888).
    pub format: u32,
    /// DRM format modifier (e.g. linear, Intel TILE_X, etc.).
    pub modifier: u64,
    pub planes: Vec<DmaPlane>,
}

/// Explicit synchronization information for a single frame. The Wayland
/// backend applies these timeline points via wp_linux_drm_syncobj_surface_v1
/// at surface commit time.
///
/// The DRM syncobj timeline itself is owned by [`crate::swapchain::Swapchain`]
/// (wrapping a [`crate::drm::DrmSyncobj`]) and is imported into the compositor
/// once at startup via `wp_linux_drm_syncobj_manager_v1.import_timeline`. Thus
/// only the acquire/release points travel per-frame.
#[derive(Debug)]
pub struct ExplicitSync {
    /// Point the compositor waits on before sampling the buffer.
    pub acquire_point: u64,
    /// Point the compositor signals when it is done with the buffer.
    pub release_point: u64,
}

/// Find the first mutually supported (format, modifier) pair between the
/// compositor's advertised formats and the Vulkan device's capabilities,
/// preferring a non-linear (non-zero, non-INVALID) modifier.
///
/// * `advertised` — (fourcc, modifiers) pairs from Wayland dmabuf feedback.
/// * `vulkan_mods` — callback that returns the Vulkan-supported modifiers
///   for a given DRM fourcc code.
pub fn negotiate(
    advertised: &[(u32, Vec<u64>)],
    vulkan_mods: impl Fn(u32) -> Vec<u64>,
) -> Result<(u32, u64)> {
    // First pass: prefer a non-linear, non-INVALID modifier.
    for &(fourcc, ref wl_mods) in advertised {
        let vk_mods = vulkan_mods(fourcc);
        for &modifier in wl_mods {
            if vk_mods.contains(&modifier) && modifier != 0 && modifier != DRM_FORMAT_MODIFIER_INVALID
            {
                return Ok((fourcc, modifier));
            }
        }
    }
    // Second pass: fall back to linear (modifier 0) if nothing else works.
    for &(fourcc, ref wl_mods) in advertised {
        let vk_mods = vulkan_mods(fourcc);
        for &modifier in wl_mods {
            if vk_mods.contains(&modifier) {
                return Ok((fourcc, modifier));
            }
        }
    }
    Err(WayvekError::NoNegotiatedFormat)
}
