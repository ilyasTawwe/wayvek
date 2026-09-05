// DRM syncobj timeline primitives.
//
// The compositor (via wp_linux_drm_syncobj_manager_v1) imports a client's DRM
// syncobj timeline; the client attaches its rendering-completion sync-file onto
// a timeline point, then tells the compositor the (acquire, release) points via
// set_acquire_point/set_release_point at commit time.

use std::os::unix::io::{AsRawFd, FromRawFd, OwnedFd};

use rustix::fs::{major, minor};
use rustix::ioctl::{self, opcode, Updater};

use crate::Result;

const DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE: u32 = 1 << 0;
const DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_TIMELINE: u32 = 1 << 1;
const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE: u32 = 1 << 2;

const DRM_IOCTL_SYNCOBJ_CREATE_OP: ioctl::Opcode =
    opcode::read_write::<DrmSyncobjCreate>(b'd', 0xBF);
const DRM_IOCTL_SYNCOBJ_DESTROY_OP: ioctl::Opcode =
    opcode::read_write::<DrmSyncobjDestroy>(b'd', 0xC0);
const DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD_OP: ioctl::Opcode =
    opcode::read_write::<DrmSyncobjHandle>(b'd', 0xC1);
const DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE_OP: ioctl::Opcode =
    opcode::read_write::<DrmSyncobjHandle>(b'd', 0xC2);
const DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_OP: ioctl::Opcode =
    opcode::read_write::<DrmSyncobjTimelineWait>(b'd', 0xCA);
const DRM_IOCTL_SYNCOBJ_TRANSFER_OP: ioctl::Opcode =
    opcode::read_write::<DrmSyncobjTransfer>(b'd', 0xCC);

/// How long to wait for the compositor to release a buffer (10s) before giving
/// up and reusing it anyway.
pub const WAIT_RELEASE_TIMEOUT_NS: i64 = 10_000_000_000;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct DrmSyncobjCreate {
    handle: u32,
    flags: u32,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct DrmSyncobjDestroy {
    handle: u32,
    pad: u32,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct DrmSyncobjHandle {
    handle: u32,
    flags: u32,
    fd: i32,
    pad: u32,
    point: u64,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct DrmSyncobjTransfer {
    src_handle: u32,
    dst_handle: u32,
    src_point: u64,
    dst_point: u64,
    flags: u32,
    pad: u32,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct DrmSyncobjTimelineWait {
    handles: u64,
    points: u64,
    timeout_nsec: i64,
    count_handles: u32,
    flags: u32,
    first_signaled: u32,
    pad: u32,
    deadline_nsec: u64,
}

/// Run a type-safe rustix `_IOWR` ioctl (`Updater`) against the DRM render node.
///
/// # Safety
///
/// `value` must be a `#[repr(C)]` struct whose layout and in/out semantics match
/// the kernel-side `struct drm_syncobj_*` the `OPCODE` corresponds to. The fd
/// must be an open, valid DRM render node.
unsafe fn drm_ioctl_rw<T, const OP: ioctl::Opcode>(
    fd: &OwnedFd,
    value: &mut T,
) -> std::io::Result<()> {
    // SAFETY: as documented above; `Updater` requests `_IOWR` semantics.
    unsafe { ioctl::ioctl(fd, Updater::<OP, T>::new(value)) }
        .map_err(std::io::Error::from)
}

/// A DRM syncobj *timeline* created on the render node and shared with the
/// compositor. Stateless w.r.t. point tracking — the caller (Swapchain) is
/// responsible for tracking per-buffer release points.
pub struct DrmSyncobj {
    fd: OwnedFd,
    handle: u32,
}

impl DrmSyncobj {
    pub fn create(fd: OwnedFd) -> Result<Self> {
        let mut create = DrmSyncobjCreate {
            handle: 0,
            flags: 0,
        };
        // SAFETY: `DrmSyncobjCreate` matches `struct drm_syncobj_create`; the
        // render node fd is valid and read-write open.
        unsafe {
            drm_ioctl_rw::<DrmSyncobjCreate, DRM_IOCTL_SYNCOBJ_CREATE_OP>(&fd, &mut create)?
        };
        Ok(Self {
            fd,
            handle: create.handle,
        })
    }

    /// Export an fd for `wp_linux_drm_syncobj_manager_v1.import_timeline`.
    /// The returned fd is a dup; the caller (or libwayland) owns it.
    pub fn export_fd(&self) -> Result<OwnedFd> {
        let mut hand = DrmSyncobjHandle {
            handle: self.handle,
            flags: DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_TIMELINE,
            fd: -1,
            ..Default::default()
        };
        // SAFETY: `DrmSyncobjHandle` matches `struct drm_syncobj_handle`; the
        // kernel writes the exported fd into the `fd` field.
        unsafe {
            drm_ioctl_rw::<DrmSyncobjHandle, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD_OP>(
                &self.fd,
                &mut hand,
            )?
        };
        if hand.fd < 0 {
            return Err(crate::WayvekError::Io(std::io::Error::other(
                "syncobj HANDLE_TO_FD returned no fd",
            )));
        }
        // SAFETY: the kernel has allocated a fresh fd>=0; we take ownership.
        Ok(unsafe { OwnedFd::from_raw_fd(hand.fd) })
    }

    /// Import a sync-file fd (the Vulkan fence) into a temporary syncobj and
    /// transfer its fence onto our timeline at the next point. Returns the
    /// acquire point for this commit.
    pub fn import_sync_file(&self, syncfile_fd: &OwnedFd) -> Result<u64> {
        let acquire_point = self.next_point();

        // Create a throwaway binary syncobj to receive the imported sync-file.
        let mut create = DrmSyncobjCreate {
            handle: 0,
            flags: 0,
        };
        // SAFETY: as `create()` above; fresh temp syncobj handle is written out.
        unsafe {
            drm_ioctl_rw::<DrmSyncobjCreate, DRM_IOCTL_SYNCOBJ_CREATE_OP>(&self.fd, &mut create)?
        };

        // Import the sync-file into that existing temp syncobj.
        let mut import = DrmSyncobjHandle {
            handle: create.handle,
            flags: DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE,
            fd: syncfile_fd.as_raw_fd(),
            ..Default::default()
        };
        // SAFETY: `DrmSyncobjHandle` matches `struct drm_syncobj_handle`; import
        // semantics are write-only from the kernel's perspective.
        unsafe {
            drm_ioctl_rw::<DrmSyncobjHandle, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE_OP>(
                &self.fd,
                &mut import,
            )?
        };
        let temp = create.handle;

        // Transfer that syncobj's fence onto the timeline at the acquire point.
        let mut transfer = DrmSyncobjTransfer {
            src_handle: temp,
            dst_handle: self.handle,
            src_point: 0,
            dst_point: acquire_point,
            flags: 0,
            pad: 0,
        };
        // SAFETY: `DrmSyncobjTransfer` matches `struct drm_syncobj_transfer`.
        unsafe {
            drm_ioctl_rw::<DrmSyncobjTransfer, DRM_IOCTL_SYNCOBJ_TRANSFER_OP>(
                &self.fd,
                &mut transfer,
            )?
        };

        // Best-effort cleanup of the temp handle after the transfer.
        let mut destroy = DrmSyncobjDestroy {
            handle: temp,
            pad: 0,
        };
        // SAFETY: `DrmSyncobjDestroy` matches `struct drm_syncobj_destroy`.
        let _ = unsafe {
            drm_ioctl_rw::<DrmSyncobjDestroy, DRM_IOCTL_SYNCOBJ_DESTROY_OP>(&self.fd, &mut destroy)
        };

        Ok(acquire_point)
    }

    /// Block until the compositor has signalled the given point, i.e. it no
    /// longer needs the buffer we handed it. Returns immediately if the point
    /// is already available.
    pub fn wait_available(&self, point: u64, timeout_nsec: i64) -> Result<()> {
        let mut handle = self.handle;
        let mut point_val = point;
        let mut wait = DrmSyncobjTimelineWait {
            handles: &mut handle as *mut u32 as u64,
            points: &mut point_val as *mut u64 as u64,
            timeout_nsec,
            count_handles: 1,
            flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE,
            ..Default::default()
        };
        // SAFETY: `handles`/`points` point at valid, aligned local variables the
        // kernel reads; `DrmSyncobjTimelineWait` layout matches the kernel struct.
        unsafe {
            drm_ioctl_rw::<DrmSyncobjTimelineWait, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_OP>(
                &self.fd,
                &mut wait,
            )?;
        }
        Ok(())
    }

    /// Return the next timeline point to use (current point + 1).
    fn next_point(&self) -> u64 {
        // We use a simple counter derived from the handle's address-space
        // uniqueness combined with a monotonically increasing point.
        // Since we are stateless, we rely on the caller to provide the point.
        // For import_sync_file, we need a point. We use a process-global counter.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        COUNTER.fetch_add(2, Ordering::Relaxed)
    }
}

impl Drop for DrmSyncobj {
    fn drop(&mut self) {
        let mut destroy = DrmSyncobjDestroy {
            handle: self.handle,
            pad: 0,
        };
        // SAFETY: `DrmSyncobjDestroy` matches `struct drm_syncobj_destroy`;
        // best-effort cleanup; errors are intentionally ignored in Drop.
        let _ = unsafe {
            drm_ioctl_rw::<DrmSyncobjDestroy, DRM_IOCTL_SYNCOBJ_DESTROY_OP>(
                &self.fd,
                &mut destroy,
            )
        };
    }
}

/// Open the DRM render node `/dev/dri/renderD<minor>` for the given dev_t, so
/// we can drive syncobj ioctls on the same GPU as the Vulkan device.
pub fn open_render_node(dev: u64) -> Result<OwnedFd> {
    let (_, minor) = (major(dev), minor(dev));
    rustix::fs::open(
        format!("/dev/dri/renderD{minor}"),
        rustix::fs::OFlags::RDWR,
        rustix::fs::Mode::empty(),
    )
    .map_err(Into::into)
}
