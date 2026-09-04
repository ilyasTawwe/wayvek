use std::ffi::CStr;
use std::os::raw::c_char;
use std::os::unix::io::{AsFd, AsRawFd, FromRawFd, OwnedFd};

use ash::vk;
use ash::{Entry, Instance};
use drm_fourcc::DrmFourcc;
use wayland_client::protocol::{wl_buffer, wl_compositor, wl_registry, wl_surface};
use wayland_client::{event_created_child, Connection, Dispatch, Proxy, QueueHandle};
use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_buffer_params_v1, zwp_linux_dmabuf_feedback_v1, zwp_linux_dmabuf_v1,
};
use wayland_protocols::wp::linux_drm_syncobj::v1::client::{
    wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1,
    wp_linux_drm_syncobj_surface_v1::WpLinuxDrmSyncobjSurfaceV1,
    wp_linux_drm_syncobj_timeline_v1::WpLinuxDrmSyncobjTimelineV1,
};
use wayland_protocols::xdg::decoration::zv1::client::{
    zxdg_decoration_manager_v1, zxdg_toplevel_decoration_v1,
};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

use rustix::fs::{major, minor};
use rustix::ioctl::{self, opcode, Updater};
use rustix::mm::{mmap, munmap, MapFlags, ProtFlags};
use zerocopy::FromBytes;

// ---------------- DRM syncobj (linux_drm_syncobj explicit sync) -----------
// The compositor (via wp_linux_drm_syncobj_manager_v1) imports a client's DRM
// syncobj timeline; the client attaches its rendering-completion sync-file onto
// a timeline point, then tells the compositor the (acquire, release) points via
// set_acquire_point/set_release_point at commit time. Mirrors egl-wayland2's
// wayland-timeline.c.

const DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE: u32 = 1 << 0;
const DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_TIMELINE: u32 = 1 << 1;
const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE: u32 = 1 << 2;

// Type-safe rustix ioctl opcodes for the DRM syncobj API on the render node.
// Each opcode is derived from the classic `_IOWR` macro (group 'd'/0x64); the
// data size is computed by rustix from the size of the #[repr(C)] struct, and
// the `_IOWR` direction matches the kernel's read-write struct semantics. These
// replace the previous manual `ioc`/`iowr` bit-shifting code.
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
/// up and reusing it anyway. The compositor should signal promptly.
const WAIT_RELEASE_TIMEOUT_NS: i64 = 10_000_000_000;

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
/// must be an open, valid DRM render node. The kernel only ever reads/writes
/// within `size_of::<T>()` bytes, so this is sound as long as `T` matches the
/// ioctl contract.
unsafe fn drm_ioctl_rw<T, const OP: ioctl::Opcode>(
    fd: &OwnedFd,
    value: &mut T,
) -> std::io::Result<()> {
    // SAFETY: as documented above; `Updater` requests `_IOWR` semantics.
    unsafe { ioctl::ioctl(fd, Updater::<OP, T>::new(value)) }
        .map_err(std::io::Error::from)
}

/// A DRM syncobj *timeline* created on the render node and shared with the
/// compositor. The `handle` is the client's view; `import_timeline` gives the
/// compositor an fd to the same underlying timeline.
struct DrmSyncobj {
    fd: OwnedFd,
    handle: u32,
    /// Last timeline point our rendering fence was attached to (the acquire
    /// point of the most recent commit).
    point: u64,
    /// Release point of the most recent commit: the point the compositor will
    /// signal when it is done with the buffer. We wait on it before reusing.
    pending_release: Option<u64>,
}

impl DrmSyncobj {
    fn create(fd: OwnedFd) -> std::io::Result<Self> {
        let mut create = DrmSyncobjCreate {
            handle: 0,
            flags: 0,
        };
        // SAFETY: `DrmSyncobjCreate` matches `struct drm_syncobj_create`; the
        // render node fd is valid and read-write open.
        unsafe { drm_ioctl_rw::<DrmSyncobjCreate, DRM_IOCTL_SYNCOBJ_CREATE_OP>(&fd, &mut create)? };
        Ok(Self {
            fd,
            handle: create.handle,
            point: 0,
            pending_release: None,
        })
    }

    /// Export an fd for `wp_linux_drm_syncobj_manager_v1.import_timeline`.
    /// The returned fd is a dup; the caller (or libwayland) owns it.
    fn import_handle_fd(&self) -> std::io::Result<OwnedFd> {
        let mut hand = DrmSyncobjHandle {
            handle: self.handle,
            flags: DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_TIMELINE,
            fd: -1,
            ..Default::default()
        };
        // SAFETY: `DrmSyncobjHandle` matches `struct drm_syncobj_handle`; the
        // kernel writes the exported fd into the `fd` field.
        unsafe { drm_ioctl_rw::<DrmSyncobjHandle, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD_OP>(&self.fd, &mut hand)? };
        if hand.fd < 0 {
            return Err(std::io::Error::other("syncobj HANDLE_TO_FD gave no fd"));
        }
        // SAFETY: the kernel has allocated a fresh fd>=0; we take ownership.
        Ok(unsafe { OwnedFd::from_raw_fd(hand.fd) })
    }

    /// Import a sync-file fd (the Vulkan fence) into a temporary syncobj and
    /// transfer its fence onto our timeline at the next point. Returns the
    /// acquire point for this commit (the point where our fence now lives), and
    /// records the following point as the release point to wait on for reuse.
    fn attach_sync_file(&mut self, syncfile_fd: &OwnedFd) -> std::io::Result<u64> {
        let acquire_point = self.point + 1;

        // Create a throwaway binary syncobj to receive the imported sync-file.
        let mut create = DrmSyncobjCreate { handle: 0, flags: 0 };
        // SAFETY: as `create()` above; fresh temp syncobj handle is written out.
        unsafe {
            drm_ioctl_rw::<DrmSyncobjCreate, DRM_IOCTL_SYNCOBJ_CREATE_OP>(&self.fd, &mut create)
                .map_err(|e| std::io::Error::new(e.kind(), format!("create temp syncobj: {e}")))?
        };

        // Import the sync-file into that existing temp syncobj. The kernel
        // resolves `handle` as the DESTINATION object for the import, so it must
        // already exist (mirrors libdrm drmSyncobjImportSyncFile / egl-wayland2).
        let mut import = DrmSyncobjHandle {
            handle: create.handle,
            flags: DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE,
            fd: syncfile_fd.as_raw_fd(),
            ..Default::default()
        };
        // SAFETY: `DrmSyncobjHandle` matches `struct drm_syncobj_handle`; import
        // semantics are write-only from the kernel's perspective.
        unsafe {
            drm_ioctl_rw::<DrmSyncobjHandle, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE_OP>(&self.fd, &mut import)
                .map_err(|e| std::io::Error::new(e.kind(), format!("import sync-file fd_to_handle: {e}")))?
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
            drm_ioctl_rw::<DrmSyncobjTransfer, DRM_IOCTL_SYNCOBJ_TRANSFER_OP>(&self.fd, &mut transfer)
                .map_err(|e| std::io::Error::new(e.kind(), format!("timeline transfer: {e}")))?
        };

        let mut destroy = DrmSyncobjDestroy {
            handle: temp,
            pad: 0,
        };
        // SAFETY: `DrmSyncobjDestroy` matches `struct drm_syncobj_destroy`;
        // best-effort cleanup of the temp handle after the transfer.
        let _ = unsafe {
            drm_ioctl_rw::<DrmSyncobjDestroy, DRM_IOCTL_SYNCOBJ_DESTROY_OP>(&self.fd, &mut destroy)
        };

        // The compositor signals the next point when it is done with the buffer.
        self.point = acquire_point;
        self.pending_release = Some(acquire_point + 1);
        Ok(acquire_point)
    }

    /// Block until the compositor has signalled the given release point, i.e.
    /// it no longer needs the buffer we handed it. Returns immediately if the
    /// point is already available.
    fn wait_release(&self, point: u64, timeout_nsec: i64) -> std::io::Result<()> {
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
        unsafe { drm_ioctl_rw::<DrmSyncobjTimelineWait, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_OP>(&self.fd, &mut wait) }
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
            drm_ioctl_rw::<DrmSyncobjDestroy, DRM_IOCTL_SYNCOBJ_DESTROY_OP>(&self.fd, &mut destroy)
        };
    }
}

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

// Window color as normalized floats (0.0-1.0) for vkCmdClearColorImage.
const CLEAR_COLOR: [f32; 4] = [0.18, 0.18, 0.64, 1.0];

// ------------------------- Format/modifier table --------------------------

/// One entry of the linux-dmabuf format table: a `(u32 format, u32 padding,
/// u64 modifier)`, tightly packed in native endianness.
#[derive(
    zerocopy::FromBytes, zerocopy::KnownLayout, zerocopy::Immutable, Copy, Clone,
)]
#[repr(C)]
struct FormatTableEntry {
    format: u32,
    _pad: u32,
    modifier: u64,
}

/// A memory-mapped, zero-copy view of the compositor's linux-dmabuf format
/// table. The mmap'd fd bytes are interpreted directly as `&[FormatTableEntry]`
/// without any per-entry allocation or copying.
struct DrmFormatTable {
    ptr: *mut std::ffi::c_void,
    size: usize,
    entries: &'static [FormatTableEntry],
}

impl DrmFormatTable {
    /// Map the fd and expose its (format, modifier) entries as a packed slice.
    /// Returns None on error.
    fn map(fd: OwnedFd, size: u32) -> Option<Self> {
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
        // slice, with no per-entry allocation or decoding loop. Returns None if
        // the length is not a multiple of the entry size.
        let entries = <[FormatTableEntry]>::ref_from_bytes_with_elems(bytes, count).ok()?;
        Some(Self {
            ptr,
            size: size as usize,
            entries,
        })
    }

    fn get(&self, index: usize) -> Option<(u32, u64)> {
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

// ---------------------------- DRM format info ----------------------------

/// One DRM format, keeping both sides' information. Only the modifiers that
/// are supported by both the compositor (Wayland) and the physical device
/// (Vulkan) are stored, in the order the compositor advertised them.
struct DrmFormatInfo {
    code: DrmFourcc,
    /// Modifiers offered by the compositor, in feedback tranche order.
    wayland_modifiers: Vec<u64>,
    /// The intersection of Wayland and Vulkan modifiers, each with the Vulkan
    /// tiling/plane properties, in Wayland order.
    available: Vec<vk::DrmFormatModifierPropertiesEXT>,
}

// ------------------------- Wayland state -------------------------------

struct State {
    running: bool,
    compositor: Option<wl_compositor::WlCompositor>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    decoration_manager: Option<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1>,
    surface: Option<wl_surface::WlSurface>,
    xdg_surface: Option<xdg_surface::XdgSurface>,
    toplevel: Option<xdg_toplevel::XdgToplevel>,
    decoration: Option<zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1>,
    // The compositor's preferred main DRM device (dev_t major, minor), learnt
    // from the linux-dmabuf feedback. Used to select the matching Vulkan GPU.
    dmabuf: Option<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1>,
    feedback: Option<zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1>,
    main_device: Option<u64>,
    // Format/modifier table mmap'd from the feedback's format_table event, and
    // the accumulated (format, modifier) pairs from the feedback tranches.
    format_table: Option<DrmFormatTable>,
    dmabuf_formats: Vec<DrmFormatInfo>,
    // Whether the feedback done event and the modifier intersection have been
    // printed, to avoid re-printing on every render.
    modifiers_printed: bool,
    // Current window size in surface coordinates.
    size: (u32, u32),
    // Vulkan renderer, initialized once the size is known.
    vk: Option<Vulkan>,
    // Queue handle used to create wayland buffer objects at present time.
    qh: Option<QueueHandle<Self>>,
    // Size (in pixels) of the buffer currently being created, for attach.
    create_size: Option<(i32, i32)>,
    // Explicit synchronization (wp_linux_drm_syncobj_v1): the compositor-side
    // manager, the per-surface extension, and the imported timeline shared with
    // our Vulkan DRM syncobj. These are mandatory (no fallback).
    syncobj_manager: Option<WpLinuxDrmSyncobjManagerV1>,
    surface_sync: Option<WpLinuxDrmSyncobjSurfaceV1>,
    syncobj_timeline: Option<WpLinuxDrmSyncobjTimelineV1>,
    // Timeline points (acquire, release) for the buffer currently being created;
    // sent via set_acquire_point/set_release_point before the surface commit.
    pending_acquire: Option<u64>,
    pending_release: Option<u64>,
}

impl State {
    /// Create the surface + xdg toplevel once all needed globals are bound.
    fn init_window(&mut self, qh: &QueueHandle<Self>) {
        if self.surface.is_some() {
            return;
        }
        let (Some(compositor), Some(wm_base)) = (&self.compositor, &self.wm_base) else {
            return;
        };

        let surface = compositor.create_surface(qh, AppData);
        let xdg_surface = wm_base.get_xdg_surface(&surface, qh, AppData);
        let toplevel = xdg_surface.get_toplevel(qh, AppData);
        toplevel.set_title(String::from("wayvek"));
        toplevel.set_app_id(String::from("wayvek"));
        toplevel.set_min_size(1, 1);

        // Request server-side window decorations if the compositor offers them.
        let decoration = self.decoration_manager.as_ref().map(|m| {
            let deco = m.get_toplevel_decoration(&toplevel, qh, AppData);
            deco.set_mode(zxdg_toplevel_decoration_v1::Mode::ServerSide);
            deco
        });

        self.surface = Some(surface.clone());
        self.xdg_surface = Some(xdg_surface);
        self.toplevel = Some(toplevel);
        self.decoration = decoration;

        // Explicit synchronization is mandatory: associate a per-surface
        // wp_linux_drm_syncobj_surface with our wl_surface. Its set_acquire_point
        // and set_release_point requests are applied at the surface commit.
        let (Some(manager), Some(surf)) = (&self.syncobj_manager, self.surface.as_ref()) else {
            panic!("wp_linux_drm_syncobj_manager_v1 is not available");
        };
        let surface_sync = manager.get_surface(surf, qh, AppData);
        self.surface_sync = Some(surface_sync);

        // Initial commit so the shell configures the window.
        surface.commit();
    }

    /// Lazily (re)create a drm-backed buffer, render a cleared frame into it,
    /// and hand the exported dmabuf to the compositor as a wl_buffer. The CPU
    /// is throttled inside [`Vulkan::present`] by waiting on the DRM syncobj
    /// timeline release point of the buffer being reused; no separate
    /// `params_pending` flag is needed.
    fn render(&mut self, width: u32, height: u32) {
        // Initialise the Vulkan renderer on first render.
        if self.vk.is_none() {
            let Some(main_device) = self.main_device else {
                eprintln!("render skipped: no main DRM device yet");
                return;
            };
            match Vulkan::new(main_device) {
                Ok(mut vk) => {
                    // Once, when both the feedback tranches and the Vulkan
                    // device are available: compute and print the intersection,
                    // then pick the format/modifier for the buffer.
                    if !self.modifiers_printed && !self.dmabuf_formats.is_empty() {
                        vk.compute_intersection(&mut self.dmabuf_formats);
                        vk.print_formats(&self.dmabuf_formats);
                        self.modifiers_printed = true;
                    }
                    vk.set_buffer_format(&self.dmabuf_formats);
                    self.vk = Some(vk);
                }
                Err(e) => {
                    eprintln!("vulkan init failed: {e}");
                    return;
                }
            }
        }

        // Import our DRM syncobj timeline into the compositor once so we can
        // set acquire/release points on it. Explicit sync is mandatory.
        if self.syncobj_timeline.is_none() {
            let (Some(manager), Some(qh2)) = (&self.syncobj_manager.clone(), self.qh.clone())
            else {
                panic!("wp_linux_drm_syncobj_manager_v1 is not available");
            };
            let import_fd = match self.vk.as_ref().and_then(|vk| vk.syncobj_import_fd()) {
                Some(fd) => fd,
                None => {
                    eprintln!("render skipped: no DRM syncobj to import");
                    return;
                }
            };
            let timeline = manager.import_timeline(import_fd.as_fd(), &qh2, AppData);
            self.syncobj_timeline = Some(timeline);
        }

        let (Some(dmabuf), Some(qh)) = (&self.dmabuf.clone(), self.qh.clone()) else {
            return;
        };

        let Some(vk) = self.vk.as_mut() else {
            eprintln!("render skipped: Vulkan renderer not initialized");
            return;
        };
        let presented = match vk.present(width, height) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("vulkan present failed: {e}");
                return;
            }
        };

        // The compositor waits on the acquire point before sampling the buffer
        // and signals the release point when it is done; both are sent on the
        // surface sync object at commit time (in the Created handler).
        self.pending_acquire = Some(presented.acquire_point);
        self.pending_release = Some(presented.release_point);

        // Wrap the exported dmabuf into a wl_buffer via linux-dmabuf params.
        let params = dmabuf.create_params(&qh, AppData);
        for (idx, plane) in presented.planes.iter().enumerate() {
            params.add(
                presented.fd.as_fd(),
                idx as u32,
                plane.offset as u32,
                plane.stride as u32,
                (presented.modifier >> 32) as u32,
                (presented.modifier & 0xffff_ffff) as u32,
            );
        }
        params.create(
            presented.width,
            presented.height,
            presented.format_fourcc,
            zwp_linux_buffer_params_v1::Flags::empty(),
        );

        self.create_size = Some((presented.width, presented.height));
    }

    /// Record a (format, modifier) pair from a feedback tranche, preserving the
    /// order in which the compositor advertised them.
    fn add_wayland_modifier(&mut self, code: DrmFourcc, modifier: u64) {
        // Find the existing slot for this format, or append a new one in order.
        if let Some(fmt) = self.dmabuf_formats.iter_mut().find(|f| f.code == code) {
            if !fmt.wayland_modifiers.contains(&modifier) {
                fmt.wayland_modifiers.push(modifier);
            }
        } else {
            self.dmabuf_formats.push(DrmFormatInfo {
                code,
                wayland_modifiers: vec![modifier],
                available: Vec::new(),
            });
        }
    }
}

// --------------------------- Ash rendering ------------------------------

/// The per-plane layout (offset + stride) of an exported dmabuf, as reported
/// by vkGetImageSubresourceLayout.
#[derive(Clone, Copy)]
struct DmaPlane {
    offset: u64,
    stride: u64,
}

/// A frame's worth of GPU work: one command buffer (reused) and one fence.
/// We only ever render into a single buffer synchronously per frame.
struct FrameData {
    command_buffer: vk::CommandBuffer,
    // Signalled once the frame's GPU work is done; waited on before reuse.
    fence: vk::Fence,
}

/// One Vulkan image exported to the compositor as a DRM dmabuf.
struct DmaBuffer {
    image: vk::Image,
    memory: vk::DeviceMemory,
    fd: OwnedFd,
    planes: Vec<DmaPlane>,
    /// Track when THIS specific buffer is safe to reuse.
    last_release_point: Option<u64>,
    /// Whether this buffer has ever been presented to the compositor. Until a
    /// buffer is presented for the first time there is nothing to wait on.
    presented: bool,
}

/// A buffer handed to the Wayland layer to wrap in a wl_buffer, together with
/// the explicit-sync timeline points for this frame.
struct PresentedBuffer {
    fd: OwnedFd,
    format_fourcc: u32,
    modifier: u64,
    planes: Vec<DmaPlane>,
    width: i32,
    height: i32,
    /// (acquire, release) timeline point for this frame's buffer.
    acquire_point: u64,
    release_point: u64,
}

struct Vulkan {
    _entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    /// Exported-fd loader for VK_KHR_external_memory_fd.
    external_fd: ash::khr::external_memory_fd::Device,
    /// Exported sync-file-fd loader for VK_KHR_external_fence_fd.
    external_fence: ash::khr::external_fence_fd::Device,
    /// DRM render node + syncobj timeline for linux_drm_syncobj explicit sync.
    syncobj: Option<DrmSyncobj>,
    /// Format and modifier of the current buffer, and its size in pixels.
    vk_format: vk::Format,
    drm_format: u32,
    modifier: u64,
    extent: vk::Extent2D,
    command_pool: vk::CommandPool,
    frame: Option<FrameData>,
    /// Double-buffered DMA buffers for the compositor.
    buffers: [Option<DmaBuffer>; 2],
    /// Which buffer slot we are currently using (0 or 1).
    current_idx: usize,
    /// Chosen (drm fourcc, vk format, modifier), set once from the intersection.
    buffer_fmt: Option<(u32, vk::Format, u64)>,
}

impl Vulkan {
    fn new(main_device: u64) -> Result<Self, String> {
        // SAFETY: all ash entry points here require an initialized `Entry` and
        // `Instance`; `create_instance`/`create_device` return valid handles on
        // success, and the loaders are constructed from those valid handles. On
        // any error we propagate without leaking handles (each step owns its
        // dependencies, which are dropped when `Self` construction short-circuits).
        unsafe {
            let entry = Entry::load().map_err(|e| e.to_string())?;
            let instance = create_instance(&entry)?;

            let (physical_device, queue_family) =
                pick_physical_device(&instance, main_device)?;

            // Pick the first queue family that supports graphics.
            let device = create_device(&instance, physical_device, queue_family)?;
            let queue = device.get_device_queue(queue_family, 0);

            let external_fd =
                ash::khr::external_memory_fd::Device::new(&instance, &device);
            let external_fence =
                ash::khr::external_fence_fd::Device::new(&instance, &device);

            // Open the DRM render node that backs this physical device (matched
            // by major/minor) so we can create the syncobj timeline for explicit
            // synchronization. Non-fatal: if it fails we fall back to implicit.
            let syncobj = open_render_node(main_device)
                .and_then(DrmSyncobj::create)
                .ok();

            Ok(Self {
                _entry: entry,
                instance,
                physical_device,
                device,
                queue,
                queue_family,
                external_fd,
                external_fence,
                syncobj,
                vk_format: vk::Format::UNDEFINED,
                drm_format: 0,
                modifier: 0,
                extent: vk::Extent2D::default(),
                command_pool: vk::CommandPool::null(),
                frame: None,
                buffers: [None, None],
                current_idx: 0,
                buffer_fmt: None,
            })
        }
    }

    /// Returns the DRM modifiers the physical device supports for `format` as
    /// `VkDrmFormatModifierPropertiesEXT` values (modifier, planes, tiling).
    fn vulkan_modifiers(&self, format: vk::Format) -> Vec<vk::DrmFormatModifierPropertiesEXT> {
        // SAFETY: `get_physical_device_format_properties2` and its p_next chain
        // (`DrmFormatModifierPropertiesListEXT`) are valid two-pass queries; the
        // `mods` buffer is sized exactly to `count`, and its lifetime outlives
        // the call so the driver may write into it.
        unsafe {
            // First pass: query the count of supported modifiers.
            let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
            let mut props2 = vk::FormatProperties2 {
                // SAFETY (chain): `list` is a valid linked struct; see fn doc.
                p_next: &mut list as *mut _ as *mut std::os::raw::c_void,
                ..Default::default()
            };
            self.instance
                .get_physical_device_format_properties2(self.physical_device, format, &mut props2);

            let count = list.drm_format_modifier_count as usize;
            if count == 0 {
                return Vec::new();
            }

            // Second pass: fill in the array of modifier properties.
            let mut mods = vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
            let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
                .drm_format_modifier_properties(&mut mods);
            let mut props2 = vk::FormatProperties2 {
                // SAFETY (chain): `list` points at `mods` (valid for `count`).
                p_next: &mut list as *mut _ as *mut std::os::raw::c_void,
                ..Default::default()
            };
            self.instance
                .get_physical_device_format_properties2(self.physical_device, format, &mut props2);

            mods
        }
    }

    /// For each format, keep only the modifiers that both the compositor
    /// (Wayland) and this physical device (Vulkan) support, attaching the
    /// Vulkan properties, and preserving the Wayland modifier order.
    fn compute_intersection(&self, formats: &mut [DrmFormatInfo]) {
        for fmt in formats.iter_mut() {
            let vk_format = drm_fourcc_to_vk(fmt.code);
            let vulkan = self.vulkan_modifiers(vk_format);
            fmt.available = fmt
                .wayland_modifiers
                .iter()
                .filter_map(|modifier| {
                    vulkan
                        .iter()
                        .find(|m| m.drm_format_modifier == *modifier)
                        .copied()
                })
                .collect();
        }
    }

    /// Print each format together with its Wayland and Vulkan properties, but
    /// only for formats available (supported) on both sides.
    fn print_formats(&self, formats: &[DrmFormatInfo]) {
        println!("DRM formats supported by both Wayland and Vulkan:");
        for fmt in formats {
            if fmt.available.is_empty() {
                continue;
            }
            println!("format={}", fmt.code);
            println!("  wayland modifiers: {:016x?}", fmt.wayland_modifiers);
            for m in &fmt.available {
                println!(
                    "  vulkan modifier={:016x} planes={} tiling={:?}",
                    m.drm_format_modifier, m.drm_format_modifier_plane_count, m.drm_format_modifier_tiling_features
                );
            }
        }
    }

    /// Pick a (DRM fourcc, Vulkan format, modifier) tuple from the given
    /// intersection, preferring a real (non-linear, non-INVALID) modifier.
    fn set_buffer_format(&mut self, formats: &[DrmFormatInfo]) {
        const INVALID_MODIFIER: u64 = (1u64 << 56) - 1;
        for fmt in formats {
            let vk_format = drm_fourcc_to_vk(fmt.code);
            if vk_format == vk::Format::UNDEFINED {
                continue;
            }
            let modifier = fmt
                .available
                .iter()
                .map(|m| m.drm_format_modifier)
                .find(|m| *m != 0 && *m != INVALID_MODIFIER)
                .unwrap_or(0);
            if modifier == INVALID_MODIFIER {
                continue;
            }
            self.buffer_fmt = Some((fmt.code as u32, vk_format, modifier));
            println!("chose format={} modifier={:016x}", fmt.code, modifier);
            break;
        }
    }

    /// Render a cleared frame into the (current) drm-backed buffer and return
    /// its exported dmabuf for the Wayland layer to wrap in a wl_buffer.
    ///
    /// This is the application's heartbeat: it blocks on the DRM syncobj
    /// timeline release point of the buffer being reused, so the CPU only
    /// advances once the compositor has signalled that the previous buffer has
    /// been released. Any error (device lost, DRM node busy, timeout) is
    /// returned as `Err` so the caller can log and skip the frame gracefully.
    fn present(&mut self, width: u32, height: u32) -> Result<PresentedBuffer, String> {
        // SAFETY: this block invokes Vulkan command-buffer recording, queue
        // submission, and fd export. The device/queue/command-buffer/fence
        // handles are all valid (created in `new`/`create_buffer`); the DrmSyncobj
        // ioctls are safe per `drm_ioctl_rw`. Every call returns its error via
        // `map_err(..)?` so nothing panics on GPU device loss or a busy DRM node.
        unsafe {
            // 1. Handle Resize: Clear the pool if dimensions changed.
            if self.extent.width != width || self.extent.height != height {
                self.device
                    .device_wait_idle()
                    .map_err(|e| format!("wait idle on resize: {e}"))?;
                self.buffers = [None, None];
                self.extent = vk::Extent2D { width, height };
            }

            let (drm_format, vk_format, modifier) = self.buffer_fmt.ok_or("no DRM format")?;

            // 2. Rotate buffers (Double Buffering).
            self.current_idx = (self.current_idx + 1) % 2;

            // 3. Lazily create the buffer for this slot.
            if self.buffers[self.current_idx].is_none() {
                self.buffers[self.current_idx] =
                    Some(self.create_buffer(width, height, drm_format, vk_format, modifier)?);
            }

            let buf = self.buffers[self.current_idx]
                .as_mut()
                .ok_or("buffer slot was not created")?;

            // 4. Throttle reuse on the explicit-sync timeline. Only a buffer that
            // has previously been presented carries a release point to wait on;
            // the compositor signals it once it is done with the buffer, so
            // blocking here is the CPU heartbeat instead of a params_pending flag.
            if let Some(wait_point) = buf.last_release_point.filter(|_| buf.presented) {
                let syncobj = self.syncobj.as_ref().ok_or("no DRM syncobj for release wait")?;
                syncobj
                    .wait_release(wait_point, WAIT_RELEASE_TIMEOUT_NS)
                    .map_err(|e| format!("buffer reuse wait failed: {e}"))?;
            }

            let frame = self
                .frame
                .as_mut()
                .ok_or("frame resources were not created")?;
            let image = buf.image;
            let command_buffer = frame.command_buffer;

            // Record a command buffer that clears the image to CLEAR_COLOR.
            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| format!("begin command buffer: {e}"))?;

            let subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);

            // UNDEFINED -> TRANSFER_DST_OPTIMAL
            let to_transfer = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(subresource)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_transfer),
            );

            let clear_value = vk::ClearColorValue { float32: CLEAR_COLOR };
            let clear_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            self.device.cmd_clear_color_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_value,
                std::slice::from_ref(&clear_range),
            );

            // TRANSFER_DST_OPTIMAL -> GENERAL (compositor reads the dmabuf).
            let to_general = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(subresource)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_general),
            );

            self.device
                .end_command_buffer(command_buffer)
                .map_err(|e| format!("end command buffer: {e}"))?;

            self.device
                .reset_fences(std::slice::from_ref(&frame.fence))
                .map_err(|e| format!("reset fence: {e}"))?;

            // Submit via the Vulkan 1.3 synchronization-2 entry point.
            let cb_info = vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer);
            let submit = vk::SubmitInfo2::default()
                .command_buffer_infos(std::slice::from_ref(&cb_info));
            self.device
                .queue_submit2(self.queue, std::slice::from_ref(&submit), frame.fence)
                .map_err(|e| format!("queue submit2: {e}"))?;

            // No CPU wait needed for the clear itself: we hand the compositor an
            // explicit acquire fence via the syncobj timeline instead.
            let get_fence_fd = vk::FenceGetFdInfoKHR::default()
                .fence(frame.fence)
                .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD);
            let syncfd = self
                .external_fence
                .get_fence_fd(&get_fence_fd)
                .map_err(|e| format!("export fence fd: {e}"))?;
            let syncfd = OwnedFd::from_raw_fd(syncfd);

            // Attach our rendering fence onto the timeline. The compositor waits
            // on the returned acquire point before sampling the buffer, and
            // signals the following release point when it is done with it.
            let syncobj = self.syncobj.as_mut().ok_or("no DRM syncobj")?;
            let acquire_point = syncobj
                .attach_sync_file(&syncfd)
                .map_err(|e| format!("attach sync-file to timeline: {e}"))?;
            let release_point = acquire_point + 1;

            // 5. Store the release point on the buffer so we know when it's
            // safe to reuse next time.
            buf.last_release_point = Some(release_point);
            buf.presented = true;

            // Hand the consumer a dup of the fd; the buffer keeps its own copy
            // for reuse on the next frame.
            let dup_fd = buf
                .fd
                .try_clone()
                .map_err(|e| format!("dup dmabuf fd: {e}"))?;

            Ok(PresentedBuffer {
                fd: dup_fd,
                format_fourcc: drm_format,
                modifier,
                planes: buf.planes.clone(),
                width: width as i32,
                height: height as i32,
                acquire_point,
                release_point,
            })
        }
    }

    /// Export the DRM syncobj timeline fd so the compositor can import it via
    /// `wp_linux_drm_syncobj_manager_v1.import_timeline`.
    fn syncobj_import_fd(&self) -> Option<OwnedFd> {
        self.syncobj.as_ref().and_then(|s| s.import_handle_fd().ok())
    }

    /// (Re)create the single drm-backed buffer at `width`x`height` using the
    /// given DRM format (fourcc), Vulkan format and modifier.
    fn create_buffer(
        &mut self,
        width: u32,
        height: u32,
        drm_format: u32,
        vk_format: vk::Format,
        modifier: u64,
    ) -> Result<DmaBuffer, String> {
        // SAFETY: creates a DRM-format-modifier image, allocates exportable
        // device memory, binds them, exports a dmabuf fd, and composes the
        // p_next chain (`external` -> `drm_info`). All Vulkan handles are valid,
        // the allocator structs are `#[repr(C)]`, and each `create_*/get_*`
        // call propagates errors via `map_err(..)?`. The exported fd is wrapped
        // in `OwnedFd` so it is not leaked.
        unsafe {
            let extent = vk::Extent2D {
                width,
                height,
            };
            let modifier_list = [modifier];
            let mut external = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let mut drm_info = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                .drm_format_modifiers(&modifier_list);

            let create_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk_format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(vk::ImageUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut external)
                .push_next(&mut drm_info);

            let image = self
                .device
                .create_image(&create_info, None)
                .map_err(|e| format!("create drm image: {e}"))?;

            // Allocate dedicated, external-exportable memory for the image.
            let memreq = self.device.get_image_memory_requirements(image);
            let memory_type_index =
                self.pick_exportable_memory_type(memreq)
                    .ok_or("no exportable device-local memory type")?;
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let mut export_info = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(memreq.size)
                .memory_type_index(memory_type_index)
                .push_next(&mut dedicated)
                .push_next(&mut export_info);
            let memory = self
                .device
                .allocate_memory(&alloc, None)
                .map_err(|e| format!("allocate external memory: {e}"))?;
            self.device
                .bind_image_memory(image, memory, 0)
                .map_err(|e| format!("bind image memory: {e}"))?;

            // Export the dmabuf fd.
            let get_fd = vk::MemoryGetFdInfoKHR::default()
                .memory(memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let fd = self
                .external_fd
                .get_memory_fd(&get_fd)
                .map_err(|e| format!("export dmabuf fd: {e}"))?;
            let fd = OwnedFd::from_raw_fd(fd);

            // Query the per-plane (here: single-plane) memory layout. DRM
            // modifier images use per-memory-plane aspects.
            let subresource = vk::ImageSubresource::default()
                .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT)
                .mip_level(0)
                .array_layer(0);
            let layout = self.device.get_image_subresource_layout(image, subresource);
            let planes = vec![DmaPlane {
                offset: layout.offset,
                stride: layout.row_pitch,
            }];

            self.vk_format = vk_format;
            self.drm_format = drm_format;
            self.modifier = modifier;
            self.buffer_fmt = Some((drm_format, vk_format, modifier));
            self.extent = extent;

            // Set up the (single) frame command buffer + fence on first use.
            if self.frame.is_none() {
                if self.command_pool == vk::CommandPool::null() {
                    let pool_info = vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.queue_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
                    self.command_pool = self
                        .device
                        .create_command_pool(&pool_info, None)
                        .map_err(|e| format!("create command pool: {e}"))?;
                }
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let bufs = self
                    .device
                    .allocate_command_buffers(&alloc_info)
                    .map_err(|e| format!("allocate command buffer: {e}"))?;
                let mut export_fence = vk::ExportFenceCreateInfo::default()
                    .handle_types(vk::ExternalFenceHandleTypeFlags::SYNC_FD);
                let fence_info = vk::FenceCreateInfo::default()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .push_next(&mut export_fence);
                let fence = self
                    .device
                    .create_fence(&fence_info, None)
                    .map_err(|e| format!("create frame fence: {e}"))?;
                self.frame = Some(FrameData {
                    command_buffer: bufs[0],
                    fence,
                });
            }

            Ok(DmaBuffer {
                image,
                memory,
                fd,
                planes,
                last_release_point: None,
                presented: false,
            })
        }
    }

    /// A memory type that is device-local and included in `req.memory_type_bits`.
    fn pick_exportable_memory_type(&self, req: vk::MemoryRequirements) -> Option<u32> {
        // SAFETY: `get_physical_device_memory_properties` returns a valid
        // `memory_type_count` and `memory_types` array owned by the driver for
        // the lifetime of the physical device; indexing within the count is safe.
        unsafe {
            let props = self
                .instance
                .get_physical_device_memory_properties(self.physical_device);
            for (i, mt) in props
                .memory_types
                .iter()
                .enumerate()
                .take(props.memory_type_count as usize)
            {
                if (req.memory_type_bits & (1 << i)) != 0
                    && mt.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                {
                    return Some(i as u32);
                }
            }
            None
        }
    }

    fn destroy_frame(&mut self) {
        // SAFETY: `destroy_fence` on a valid fence handle is an allowed
        // destruction call; the fence is removed from `self.frame` first so it
        // cannot be used after destruction.
        unsafe {
            if let Some(frame) = self.frame.take() {
                self.device.destroy_fence(frame.fence, None);
            }
        }
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        // SAFETY: Destroying the logical device, images, memory, command pool,
        // and instance is the standard teardown path; all Vulkan handles are
        // valid and no other thread is using them at drop time.
        unsafe {
            // Ensure all in-flight rendering has completed before destroying.
            // Best-effort: a lost device leaves nothing to wait on, so ignore
            // the error here (destruction proceeds regardless).
            let _ = self.device.device_wait_idle();

            self.destroy_frame();
            for slot in &mut self.buffers {
                if let Some(buffer) = slot.take() {
                    self.device.destroy_image(buffer.image, None);
                    self.device.free_memory(buffer.memory, None);
                    // buffer.fd closes on drop.
                }
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_instance(entry: &Entry) -> Result<Instance, String> {
    unsafe {
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::make_api_version(0, 1, 3, 0))
            .application_name(c"wayvek")
            .application_version(1)
            .engine_name(c"wayvek")
            .engine_version(1);

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&[]);

        // SAFETY: `app_info` and `create_info` are valid Vulkan descriptions
        // with internal pointers to `c"wayvek"` (which live for 'static).
        entry
            .create_instance(&create_info, None)
            .map_err(|e| format!("create instance: {e}"))
    }
}

/// Select the physical device whose render DRM node matches the compositor's
/// preferred device, and return its graphics queue family.
///
/// This performs a single `get_physical_device_properties2` call per candidate
/// with a single `p_next` chain (`Vulkan11Properties -> DrmPropertiesEXT`) so
/// that the GPU identity and the DRM major/minor are gathered atomically, and
/// the queue-family scan is done directly on the match.
fn pick_physical_device(
    instance: &Instance,
    dev: u64,
) -> Result<(vk::PhysicalDevice, u32), String> {
    let (major, minor) = (major(dev), minor(dev));
    unsafe {
        let devices = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("enumerate physical devices: {e}"))?;

        for device in &devices {
            // Build the p_next chain in a single call: properties2 -> vulkan11
            // -> drm. The DRM extension gives us the render-node identity, and
            // Vulkan11Properties is carried along for a future-proof atomic query.
            let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
            let mut vulkan11 = vk::PhysicalDeviceVulkan11Properties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default();
            // SAFETY: each struct in the chain is `#[repr(C)]` with a matching
            // `s_type`, and the lifetimes outlast the call; we only read them
            // afterwards.
            vulkan11.p_next = &mut drm as *mut _ as *mut std::os::raw::c_void;
            props2.p_next = &mut vulkan11 as *mut _ as *mut std::os::raw::c_void;
            instance.get_physical_device_properties2(*device, &mut props2);

            let drm_matches = drm.has_render == vk::TRUE
                && drm.render_major as u32 == major
                && drm.render_minor as u32 == minor;
            if !drm_matches {
                continue;
            }

            // Find a graphics queue family on this device.
            let queue_families = instance.get_physical_device_queue_family_properties(*device);
            let queue_family = queue_families
                .iter()
                .enumerate()
                .find(|(_, qf)| qf.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|(i, _)| i as u32);
            if let Some(q) = queue_family {
                let name = cstr_to_string(props2.properties.device_name.as_ptr());
                eprintln!("using main GPU: {name}");
                return Ok((*device, q));
            }
        }

        Err(format!("no Vulkan device matched main DRM {major}:{minor}"))
    }
}

/// Open the DRM render node `/dev/dri/renderD<minor>` for the given dev_t, so
/// we can drive syncobj ioctls on the same GPU as our Vulkan device.
fn open_render_node(dev: u64) -> std::io::Result<OwnedFd> {
    let (_, minor) = (major(dev), minor(dev));
    rustix::fs::open(
        format!("/dev/dri/renderD{minor}"),
        rustix::fs::OFlags::RDWR,
        rustix::fs::Mode::empty(),
    )
    .map_err(std::io::Error::from)
}

fn create_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
) -> Result<ash::Device, String> {
    // SAFETY: `create_device` is called with a valid `Instance` and
    // `PhysicalDevice`; the queue create info and extension list are valid
    // `#[repr(C)]` structures referencing `'static` C-string extension names
    // (ash's `NAME` constants). The returned `ash::Device` is owned by this
    // function's caller.
    unsafe {
        let priority = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priority);

        let device_extensions = [
            ash::khr::external_memory_fd::NAME.as_ptr(),
            ash::ext::external_memory_dma_buf::NAME.as_ptr(),
            ash::khr::external_fence_fd::NAME.as_ptr(),
            vk::EXT_IMAGE_DRM_FORMAT_MODIFIER_NAME.as_ptr(),
        ];

        // Enable the Vulkan 1.3 synchronization2 feature so `queue_submit2` and
        // `vk::SubmitInfo2` are usable on the queue.
        let mut sync2_features = vk::PhysicalDeviceSynchronization2Features::default()
            .synchronization2(true);

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions)
            .push_next(&mut sync2_features);

        instance
            .create_device(physical_device, &create_info, None)
            .map_err(|e| format!("create device: {e}"))
    }
}

/// Map a DRM_FORMAT fourcc to the best-matching linear Vulkan format, so we
/// can query a physical device's supported modifiers for that format.
fn drm_fourcc_to_vk(fourcc: DrmFourcc) -> vk::Format {
    match fourcc {
        // XRGB8888 / ARGB8888 -> byte order B, G, R, X/A
        DrmFourcc::Xrgb8888 | DrmFourcc::Argb8888 => vk::Format::B8G8R8A8_UNORM,
        // XBGR8888 / ABGR8888 -> byte order R, G, B, X/A
        DrmFourcc::Xbgr8888 | DrmFourcc::Abgr8888 => vk::Format::R8G8B8A8_UNORM,
        // RGB565
        DrmFourcc::Rgb565 => vk::Format::R5G6B5_UNORM_PACK16,
        // NV12 (semi-planar YUV 4:2:0)
        DrmFourcc::Nv12 => vk::Format::G8_B8R8_2PLANE_420_UNORM,
        _ => vk::Format::UNDEFINED,
    }
}

/// Converts a null-terminated C string pointer (e.g. a Vulkan device name) to
/// a Rust `String`. A null pointer yields an empty string.
///
/// # Safety
///
/// `ptr` must point to a valid, null-terminated UTF-8-compatible C string, or be
/// null. Vulkan drivers provide such strings for `swapchain.pNext` device names.
unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: contract described on the function.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

// ----------------------------- Dispatch --------------------------------

struct AppData;
struct GlobalData;

impl Dispatch<wl_registry::WlRegistry, GlobalData> for State {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _data: &GlobalData,
        _conn: &Connection,
        qh: &QueueHandle<State>,
    ) {
        if let wl_registry::Event::Global { name, interface, version, .. } = event {
            match interface.as_str() {
                "wl_compositor" => {
                    // Version 4 is required so surfaces can use damage_buffer.
                    let version_used = version.min(4);
                    let proxy = registry.bind::<wl_compositor::WlCompositor, _, _>(
                        name,
                        version_used,
                        qh,
                        AppData,
                    );
                    state.compositor = Some(proxy);
                }
                "xdg_wm_base" => {
                    let proxy =
                        registry.bind::<xdg_wm_base::XdgWmBase, _, _>(name, 1, qh, AppData);
                    state.wm_base = Some(proxy);
                }
                "zxdg_decoration_manager_v1" => {
                    let proxy = registry.bind::<
                        zxdg_decoration_manager_v1::ZxdgDecorationManagerV1,
                        _,
                        _,
                    >(name, 1, qh, AppData);
                    state.decoration_manager = Some(proxy);
                }
                // Version 5 is the newest that still reports the legacy
                // zwp_linux_dmabuf_feedback_v1.main_device event.
                "zwp_linux_dmabuf_v1" => {
                    let proxy =
                        registry.bind::<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1, _, _>(
                            name,
                            5,
                            qh,
                            AppData,
                        );
                    let feedback = proxy.get_default_feedback(qh, AppData);
                    state.dmabuf = Some(proxy);
                    state.feedback = Some(feedback);
                }
                "wp_linux_drm_syncobj_manager_v1" => {
                    let proxy = registry.bind::<WpLinuxDrmSyncobjManagerV1, _, _>(name, 1, qh, AppData);
                    state.syncobj_manager = Some(proxy);
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, AppData> for State {
    fn event(
        _state: &mut Self,
        proxy: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            proxy.pong(serial);
        }
    }
}

impl Dispatch<wl_compositor::WlCompositor, AppData> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wl_compositor::WlCompositor,
        _event: wl_compositor::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1, AppData> for State {
    fn event(
        _state: &mut Self,
        _proxy: &zxdg_decoration_manager_v1::ZxdgDecorationManagerV1,
        _event: zxdg_decoration_manager_v1::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1, AppData> for State {
    fn event(
        _state: &mut Self,
        _proxy: &zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1,
        _event: zwp_linux_dmabuf_v1::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1, AppData> for State {
    fn event(
        state: &mut Self,
        _proxy: &zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1,
        event: zwp_linux_dmabuf_feedback_v1::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            // The main_device event reports the compositor's preferred DRM
            // device, whose dev_t is serialized into the byte array.
            zwp_linux_dmabuf_feedback_v1::Event::MainDevice { device } => {
                let mut bytes = [0u8; 8];
                let n = device.len().min(8);
                bytes[..n].copy_from_slice(&device[..n]);
                state.main_device = Some(u64::from_le_bytes(bytes));
            }
            // The compositor's packed format + modifier table as a mapped fd.
            zwp_linux_dmabuf_feedback_v1::Event::FormatTable { fd, size } => {
                state.format_table = DrmFormatTable::map(fd, size);
            }
            // Each tranche_formats event carries u16 indices into the table;
            // look each up to collect the (format, modifier) pairs.
            zwp_linux_dmabuf_feedback_v1::Event::TrancheFormats { indices } => {
                // Collect (code, modifier) pairs while the table is borrowed,
                // then update state once the borrow is released.
                let mut pairs: Vec<(DrmFourcc, u64)> = Vec::new();
                if let Some(table) = &state.format_table {
                    // Decode the packed u16 index array with zero-copy parsing.
                    for pair in indices.as_chunks::<2>().0 {
                        let Ok(idx) = u16::ref_from_bytes(pair) else { continue };
                        if let Some((format, modifier)) = table.get(*idx as usize)
                            && let Ok(code) = DrmFourcc::try_from(format)
                        {
                            pairs.push((code, modifier));
                        }
                    }
                }
                for (code, modifier) in pairs {
                    state.add_wayland_modifier(code, modifier);
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1, AppData> for State {
    fn event(
        state: &mut Self,
        params: &zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1,
        event: zwp_linux_buffer_params_v1::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            // Buffer creation succeeded: attach it to the surface and commit.
            zwp_linux_buffer_params_v1::Event::Created { buffer } => {
                params.destroy();
                if let Some(surface) = &state.surface {
                    // Explicit sync: set the acquire/release timeline points that
                    // apply to this commit, immediately before attaching and
                    // committing, per the linux_drm_syncobj protocol contract.
                    let (Some(surface_sync), Some(timeline)) =
                        (&state.surface_sync, &state.syncobj_timeline)
                    else {
                        eprintln!("explicit sync objects missing; dropping buffer");
                        state.create_size = None;
                        return;
                    };
                    let acquire = match state.pending_acquire.take() {
                        Some(a) => a,
                        None => {
                            eprintln!("no pending acquire point; dropping buffer");
                            state.create_size = None;
                            return;
                        }
                    };
                    let release = match state.pending_release.take() {
                        Some(r) => r,
                        None => {
                            eprintln!("no pending release point; dropping buffer");
                            state.create_size = None;
                            return;
                        }
                    };
                    surface_sync.set_acquire_point(timeline, (acquire >> 32) as u32, acquire as u32);
                    surface_sync.set_release_point(timeline, (release >> 32) as u32, release as u32);

                    surface.attach(Some(&buffer), 0, 0);
                    let (w, h) = state.create_size.unwrap_or((0, 0));
                    surface.damage_buffer(0, 0, w, h);
                    surface.commit();
                }
                state.create_size = None;
            }
            // Creation failed: try again on the next configure.
            zwp_linux_buffer_params_v1::Event::Failed => {
                params.destroy();
                eprintln!("linux-dmabuf buffer creation failed");
                state.create_size = None;
                state.pending_acquire = None;
                state.pending_release = None;
            }
            _ => {}
        }
    }

    event_created_child!(
        State,
        zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1,
        [zwp_linux_buffer_params_v1::EVT_CREATED_OPCODE => (wl_buffer::WlBuffer, AppData)]
    );
}

impl Dispatch<wl_buffer::WlBuffer, AppData> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wl_buffer::WlBuffer,
        _event: wl_buffer::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WpLinuxDrmSyncobjManagerV1, AppData> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WpLinuxDrmSyncobjManagerV1,
        _event: <WpLinuxDrmSyncobjManagerV1 as Proxy>::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WpLinuxDrmSyncobjSurfaceV1, AppData> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WpLinuxDrmSyncobjSurfaceV1,
        _event: <WpLinuxDrmSyncobjSurfaceV1 as Proxy>::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WpLinuxDrmSyncobjTimelineV1, AppData> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WpLinuxDrmSyncobjTimelineV1,
        _event: <WpLinuxDrmSyncobjTimelineV1 as Proxy>::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1, AppData> for State {
    fn event(
        _state: &mut Self,
        _proxy: &zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1,
        _event: zxdg_toplevel_decoration_v1::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_surface::WlSurface, AppData> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wl_surface::WlSurface,
        _event: wl_surface::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<xdg_surface::XdgSurface, AppData> for State {
    fn event(
        state: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            xdg_surface.ack_configure(serial);
            let (w, h) = state.size;
            state.render(w, h);
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, AppData> for State {
    fn event(
        state: &mut Self,
        _proxy: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            xdg_toplevel::Event::Configure { width, height, .. } => {
                if width != 0 && height != 0 {
                    state.size = (width as u32, height as u32);
                }
            }
            xdg_toplevel::Event::Close => state.running = false,
            _ => {}
        }
    }
}

// -------------------------------- Main ---------------------------------

fn main() {
    let conn = Connection::connect_to_env().expect("failed to connect to Wayland display");

    let mut event_queue = conn.new_event_queue();
    let qh = event_queue.handle();

    let mut state = State {
        running: true,
        compositor: None,
        wm_base: None,
        decoration_manager: None,
        surface: None,
        xdg_surface: None,
        toplevel: None,
        decoration: None,
        dmabuf: None,
        feedback: None,
        main_device: None,
        format_table: None,
        dmabuf_formats: Vec::new(),
        modifiers_printed: false,
        size: (WINDOW_WIDTH, WINDOW_HEIGHT),
        vk: None,
        qh: Some(qh.clone()),
        create_size: None,
        syncobj_manager: None,
        surface_sync: None,
        syncobj_timeline: None,
        pending_acquire: None,
        pending_release: None,
    };

    // Roundtrip once to read the globals and create the window.
    {
        let display = conn.display();
        let _registry = display.get_registry(&qh, GlobalData);
        event_queue
            .roundtrip(&mut state)
            .expect("failed to roundtrip while reading globals");
    }

    // Create the window now that all globals are known.
    state.init_window(&qh);

    // Blocking dispatch loop until the window is closed by the compositor.
    while state.running {
        event_queue
            .blocking_dispatch(&mut state)
            .expect("error dispatching Wayland events");
    }
}
