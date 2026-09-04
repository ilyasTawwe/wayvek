use std::ffi::{CStr, c_char};
use std::os::unix::io::{FromRawFd, OwnedFd};

use ash::vk;
use ash::{Entry, Instance};
use drm_fourcc::DrmFourcc;

use crate::{DmaFrame, DmaPlane};

/// A frame's worth of GPU work: one command buffer (reused) and one fence.
struct FrameData {
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

/// One Vulkan image exported to the compositor as a DMA-BUF.
struct DmaBuffer {
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    fd: OwnedFd,
    planes: Vec<DmaPlane>,
}

impl Drop for DmaBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.free_memory(self.memory, None);
            self.device.destroy_image(self.image, None);
        }
    }
}

pub struct Vulkan {
    _entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    external_fd: ash::khr::external_memory_fd::Device,
    external_fence: ash::khr::external_fence_fd::Device,
    buffer_fmt: Option<(u32, vk::Format, u64)>,
    extent: vk::Extent2D,
    command_pool: vk::CommandPool,
    frame: Option<FrameData>,
    buffers: [Option<DmaBuffer>; 2],
}

impl Vulkan {
    pub fn new(main_device: u64) -> Result<Self, String> {
        // SAFETY: all ash entry points here require an initialized `Entry` and
        // `Instance`; `create_instance`/`create_device` return valid handles on
        // success, and the loaders are constructed from those valid handles.
        unsafe {
            let entry = Entry::load().map_err(|e| e.to_string())?;
            let instance = create_instance(&entry)?;

            let (physical_device, queue_family) =
                pick_physical_device(&instance, main_device)?;

            let device = create_device(&instance, physical_device, queue_family)?;
            let queue = device.get_device_queue(queue_family, 0);

            let external_fd =
                ash::khr::external_memory_fd::Device::new(&instance, &device);
            let external_fence =
                ash::khr::external_fence_fd::Device::new(&instance, &device);

            Ok(Self {
                _entry: entry,
                instance,
                physical_device,
                device,
                queue,
                queue_family,
                external_fd,
                external_fence,
                buffer_fmt: None,
                extent: vk::Extent2D::default(),
                command_pool: vk::CommandPool::null(),
                frame: None,
                buffers: [None, None],
            })
        }
    }

    /// Returns the DRM modifiers the physical device supports for the given
    /// DRM fourcc code.
    pub fn get_supported_modifiers(&self, fourcc: u32) -> Vec<u64> {
        let Ok(code) = DrmFourcc::try_from(fourcc) else {
            return Vec::new();
        };
        let vk_format = drm_fourcc_to_vk(code);
        if vk_format == vk::Format::UNDEFINED {
            return Vec::new();
        }
        self.vulkan_modifiers(vk_format)
            .iter()
            .map(|m| m.drm_format_modifier)
            .collect()
    }

    /// Configure which DRM format and modifier to use for buffer creation.
    /// Must be called before the first draw.
    pub fn set_buffer_format(&mut self, fourcc: u32, modifier: u64) {
        let Ok(code) = DrmFourcc::try_from(fourcc) else {
            return;
        };
        let vk_format = drm_fourcc_to_vk(code);
        self.buffer_fmt = Some((fourcc, vk_format, modifier));
    }

    /// Render a frame into the current DMA-backed buffer and return its
    /// exported DMA-BUF metadata and the GPU completion fence as a sync-file fd.
    ///
    /// The caller provides a `record` closure that receives the device,
    /// command buffer (already begun), and image handle (initially in
    /// `UNDEFINED` layout) and records GPU commands.  The renderer handles
    /// buffer creation, command-buffer begin/end, queue submission, and
    /// fence export.
    ///
    /// This method is purely GPU-side: it has no knowledge of Wayland protocols
    /// or DRM syncobj timelines. The caller (Swapchain) is responsible for
    /// backpressure (waiting on buffer release) and timeline synchronization.
    pub fn draw(
        &mut self,
        slot: usize,
        width: u32,
        height: u32,
        record: impl FnOnce(usize, ash::Device, vk::CommandBuffer, vk::Image),
    ) -> Result<(DmaFrame, OwnedFd), String> {
        // SAFETY: this block invokes Vulkan command-buffer recording, queue
        // submission, and fd export. All Vulkan handles are valid (created in
        // `new`/`create_buffer`); errors are propagated via `map_err(..)?`.
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

            // 2. Lazily create the buffer for this slot.
            if self.buffers[slot].is_none() {
                self.buffers[slot] =
                    Some(self.create_buffer(width, height, vk_format, modifier)?);
            }

            let buf = self.buffers[slot]
                .as_ref()
                .ok_or("buffer slot was not created")?;

            let frame = self
                .frame
                .as_mut()
                .ok_or("frame resources were not created")?;
            let image = buf.image;
            let command_buffer = frame.command_buffer;

            // 4. No CPU wait on the Vulkan fence: the Swapchain's DRM syncobj
            // release-point wait guarantees the previous frame's GPU work is
            // complete before we reuse this command buffer. The fence is only
            // used to export a sync-file for the compositor's acquire fence.

            // Begin command buffer and let the caller record GPU commands.
            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| format!("begin command buffer: {e}"))?;

            record(slot, self.device.clone(), command_buffer, image);

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

            // Export the completion fence as a sync-file fd.
            let get_fence_fd = vk::FenceGetFdInfoKHR::default()
                .fence(frame.fence)
                .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD);
            let syncfd = self
                .external_fence
                .get_fence_fd(&get_fence_fd)
                .map_err(|e| format!("export fence fd: {e}"))?;
            let syncfd = OwnedFd::from_raw_fd(syncfd);

            // Build the DmaFrame from the buffer's metadata.
            let frame = DmaFrame {
                fd: buf.fd.try_clone().map_err(|e| format!("dup dmabuf fd: {e}"))?,
                width,
                height,
                format: drm_format,
                modifier,
                planes: buf.planes.clone(),
            };

            Ok((frame, syncfd))
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Returns the DRM modifiers the physical device supports for `format`.
    fn vulkan_modifiers(
        &self,
        format: vk::Format,
    ) -> Vec<vk::DrmFormatModifierPropertiesEXT> {
        // SAFETY: two-pass Vulkan format query; p_next chain is valid,
        // `mods` buffer is sized to `count`, lifetimes outlast the calls.
        unsafe {
            let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
            let mut props2 = vk::FormatProperties2 {
                p_next: &mut list as *mut _ as *mut std::os::raw::c_void,
                ..Default::default()
            };
            self.instance
                .get_physical_device_format_properties2(self.physical_device, format, &mut props2);

            let count = list.drm_format_modifier_count as usize;
            if count == 0 {
                return Vec::new();
            }

            let mut mods = vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
            let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
                .drm_format_modifier_properties(&mut mods);
            let mut props2 = vk::FormatProperties2 {
                p_next: &mut list as *mut _ as *mut std::os::raw::c_void,
                ..Default::default()
            };
            self.instance
                .get_physical_device_format_properties2(self.physical_device, format, &mut props2);

            mods
        }
    }

    /// Create the single DMA-backed buffer at `width`x`height`.
    fn create_buffer(
        &mut self,
        width: u32,
        height: u32,
        vk_format: vk::Format,
        modifier: u64,
    ) -> Result<DmaBuffer, String> {
        // SAFETY: creates a DRM-format-modifier image, allocates exportable
        // device memory, binds them, exports a dmabuf fd, and composes the
        // p_next chain. All Vulkan handles are valid, the allocator structs
        // are `#[repr(C)]`, and errors are propagated via `map_err(..)?`.
        unsafe {
            let extent = vk::Extent2D { width, height };
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

            let memreq = self.device.get_image_memory_requirements(image);
            let memory_type_index = self
                .pick_exportable_memory_type(memreq)
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

            let get_fd = vk::MemoryGetFdInfoKHR::default()
                .memory(memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let fd = self
                .external_fd
                .get_memory_fd(&get_fd)
                .map_err(|e| format!("export dmabuf fd: {e}"))?;
            let fd = OwnedFd::from_raw_fd(fd);

            let subresource = vk::ImageSubresource::default()
                .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT)
                .mip_level(0)
                .array_layer(0);
            let layout = self.device.get_image_subresource_layout(image, subresource);
            let planes = vec![DmaPlane {
                offset: layout.offset,
                stride: layout.row_pitch,
            }];

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
                device: self.device.clone(),
                image,
                memory,
                fd,
                planes,
            })
        }
    }

    /// A memory type that is device-local and in the required type bits.
    fn pick_exportable_memory_type(&self, req: vk::MemoryRequirements) -> Option<u32> {
        // SAFETY: `get_physical_device_memory_properties` returns a valid
        // `memory_type_count` and `memory_types` array.
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
                    && mt.property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                {
                    return Some(i as u32);
                }
            }
            None
        }
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        // SAFETY: Standard teardown path; all Vulkan handles are valid and
        // no other thread is using them at drop time.
        unsafe {
            let _ = self.device.device_wait_idle();

            if let Some(frame) = self.frame.take() {
                self.device.destroy_fence(frame.fence, None);
            }
            for slot in &mut self.buffers {
                slot.take();
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ---------------------------------------------------------------------------
// Free helper functions
// ---------------------------------------------------------------------------

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

fn pick_physical_device(
    instance: &Instance,
    dev: u64,
) -> Result<(vk::PhysicalDevice, u32), String> {
    let (major, minor) = (rustix::fs::major(dev), rustix::fs::minor(dev));
    unsafe {
        let devices = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("enumerate physical devices: {e}"))?;

        for device in &devices {
            let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
            let mut vulkan11 = vk::PhysicalDeviceVulkan11Properties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default();
            // SAFETY: each struct in the chain is `#[repr(C)]` with a matching
            // `s_type`, and the lifetimes outlast the call.
            vulkan11.p_next = &mut drm as *mut _ as *mut std::os::raw::c_void;
            props2.p_next = &mut vulkan11 as *mut _ as *mut std::os::raw::c_void;
            instance.get_physical_device_properties2(*device, &mut props2);

            let drm_matches = drm.has_render == vk::TRUE
                && drm.render_major as u32 == major
                && drm.render_minor as u32 == minor;
            if !drm_matches {
                continue;
            }

            let queue_families =
                instance.get_physical_device_queue_family_properties(*device);
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

fn create_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
) -> Result<ash::Device, String> {
    // SAFETY: `create_device` is called with a valid `Instance` and
    // `PhysicalDevice`; extension names are `'static` ash constants.
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

        let mut sync2_features =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions)
            .push_next(&mut sync2_features);

        instance
            .create_device(physical_device, &create_info, None)
            .map_err(|e| format!("create device: {e}"))
    }
}

pub fn drm_fourcc_to_vk(fourcc: DrmFourcc) -> vk::Format {
    match fourcc {
        DrmFourcc::Xrgb8888 | DrmFourcc::Argb8888 => vk::Format::B8G8R8A8_UNORM,
        DrmFourcc::Xbgr8888 | DrmFourcc::Abgr8888 => vk::Format::R8G8B8A8_UNORM,
        DrmFourcc::Rgb565 => vk::Format::R5G6B5_UNORM_PACK16,
        DrmFourcc::Nv12 => vk::Format::G8_B8R8_2PLANE_420_UNORM,
        _ => vk::Format::UNDEFINED,
    }
}

unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: contract described on the function.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}
