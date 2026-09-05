use ash::vk;
use thiserror::Error;

/// Error type for the whole wayvek library. Every failure path funnels into a
/// single typed error so callers can `?` it into their own error type (e.g.
/// anyhow) while still being able to match on categories.
#[derive(Debug, Error)]
pub enum WayvekError {
    /// A Vulkan API call failed.
    #[error("vulkan: {0}")]
    Vulkan(#[from] vk::Result),

    /// A POSIX / ioctl / file-descriptor operation failed.
    #[error("{0}")]
    Io(#[from] std::io::Error),

    /// The Vulkan loader could not be opened.
    #[error("failed to load Vulkan library: {0}")]
    LoadLibrary(#[from] ash::LoadingError),

    /// No Vulkan physical device exposed the expected DRM render node.
    #[error("no Vulkan device matches main DRM {major}:{minor}")]
    NoMatchingDevice { major: u32, minor: u32 },

    /// No device-local memory type is both allocation-capable and exportable.
    #[error("no exportable device-local memory type")]
    NoExportableMemoryType,

    /// No (format, modifier) pair is supported by both the compositor and the
    /// Vulkan device.
    #[error("no mutually supported DRM format")]
    NoNegotiatedFormat,
}

impl From<rustix::io::Errno> for WayvekError {
    fn from(err: rustix::io::Errno) -> Self {
        WayvekError::Io(std::io::Error::from(err))
    }
}

/// Convenience alias used throughout the library.
pub type Result<T> = std::result::Result<T, WayvekError>;