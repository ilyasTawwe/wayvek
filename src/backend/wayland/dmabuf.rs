use std::os::unix::io::OwnedFd;

use drm_fourcc::DrmFourcc;
use rustix::mm::{mmap, munmap, MapFlags, ProtFlags};
use zerocopy::FromBytes;

/// One entry of the linux-dmabuf format table: a `(u32 format, u32 padding,
/// u64 modifier)`, tightly packed in native endianness.
#[derive(
    zerocopy::FromBytes, zerocopy::KnownLayout, zerocopy::Immutable, Copy, Clone,
)]
#[repr(C)]
pub struct FormatTableEntry {
    pub format: u32,
    pub _pad: u32,
    pub modifier: u64,
}

/// A memory-mapped, zero-copy view of the compositor's linux-dmabuf format
/// table. The mmap'd fd bytes are interpreted directly as `&[FormatTableEntry]`
/// without any per-entry allocation or copying.
pub struct DrmFormatTable {
    ptr: *mut std::ffi::c_void,
    size: usize,
    entries: &'static [FormatTableEntry],
}

impl DrmFormatTable {
    /// Map the fd and expose its (format, modifier) entries as a packed slice.
    pub fn map(fd: OwnedFd, size: u32) -> Option<Self> {
        // SAFETY: `fd` is a fresh mmap source returned by the compositor; we map
        // it read-only (MAP_PRIVATE + READ), so no aliasing writes occur. The
        // returned pointer is held by `DrmFormatTable` and munmap'd on drop, so
        // the slice never outlives the mapping.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                size as usize,
                ProtFlags::READ,
                MapFlags::PRIVATE,
                &fd,
                0,
            )
            .ok()?
        };
        // SAFETY: `ptr` is valid for `size` readable bytes (just mapped).
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, size as usize) };
        let count = bytes.len() / std::mem::size_of::<FormatTableEntry>();
        // Zero-copy: reinterpret the mmap'd bytes directly as the packed entry
        // slice, with no per-entry allocation or decoding loop.
        let entries = <[FormatTableEntry]>::ref_from_bytes_with_elems(bytes, count).ok()?;
        Some(Self {
            ptr,
            size: size as usize,
            entries,
        })
    }

    pub fn get(&self, index: usize) -> Option<(u32, u64)> {
        self.entries.get(index).map(|e| (e.format, e.modifier))
    }
}

impl Drop for DrmFormatTable {
    fn drop(&mut self) {
        // SAFETY: `ptr` (length `size`) was returned by `mmap` and is still
        // valid; we destroy the mapping while `entries` is the last live borrow.
        unsafe {
            let _ = munmap(self.ptr, self.size);
        }
    }
}

/// One DRM format with the compositor-advertised modifiers. Populated from
/// the linux-dmabuf feedback tranches. No Vulkan types — the intersection
/// is computed externally by the negotiate() utility.
pub struct FormatInfo {
    pub code: DrmFourcc,
    /// Modifiers offered by the compositor, in feedback tranche order.
    pub modifiers: Vec<u64>,
}
