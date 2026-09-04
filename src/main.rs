use std::ffi::CString;
use std::os::raw::c_char;

use ash::vk;
use ash::{Entry, Instance};
use wayland_client::backend::{Backend, ObjectId};
use wayland_client::protocol::{wl_compositor, wl_registry, wl_surface};
use wayland_client::{Connection, Dispatch, Proxy, QueueHandle};
use wayland_protocols::xdg::decoration::zv1::client::{
    zxdg_decoration_manager_v1, zxdg_toplevel_decoration_v1,
};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

// Window color as normalized floats (0.0-1.0) for vkCmdClearColorImage.
const CLEAR_COLOR: [f32; 4] = [0.18, 0.18, 0.64, 1.0];

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
    // Current window size in surface coordinates.
    size: (u32, u32),
    // Vulkan renderer, initialized once the surface + size are known.
    vk: Option<Vulkan>,
    // Raw pointers handed to the Vulkan wayland surface. Kept alive here so the
    // borrow checker sees they outlive the swapchain.
    _surface_ptr: Option<*mut vk::wl_surface>,
    _display_ptr: Option<*mut vk::wl_display>,
}

impl State {
    /// Create the surface + xdg toplevel once all needed globals are bound.
    fn init_window(&mut self, qh: &QueueHandle<Self>, display_ptr: *mut vk::wl_display) {
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
        self._display_ptr = Some(display_ptr);

        // The Vulkan WSI will attach + commit dmabufs for us, but we still do an
        // initial commit so the shell configures the window.
        surface.commit();
    }

    /// Lazily (re)create the swapchain and render a cleared frame.
    fn render(&mut self, width: u32, height: u32) {
        let (Some(surface), Some(display_ptr)) = (&self.surface, &self._display_ptr) else {
            return;
        };

        // Initialise the Vulkan renderer on first render.
        if self.vk.is_none() {
            let surface_ptr = surface.id().as_ptr() as *mut vk::wl_surface;
            self._surface_ptr = Some(surface_ptr);
            let display_ptr = *display_ptr;
            match Vulkan::new(display_ptr, surface_ptr) {
                Ok(vk) => self.vk = Some(vk),
                Err(e) => {
                    eprintln!("vulkan init failed: {e}");
                    return;
                }
            }
        }

        let vk = self.vk.as_mut().unwrap();
        if let Err(e) = vk.present(width, height) {
            eprintln!("vulkan present failed: {e}");
        }
    }
}

// --------------------------- Ash rendering ------------------------------

struct Vulkan {
    _entry: Entry,
    instance: Instance,
    surface: vk::SurfaceKHR,
    surface_fn: ash::khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    swapchain: ash::khr::swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    format: vk::Format,
    extent: vk::Extent2D,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    acquire_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
}

impl Vulkan {
    fn new(display_ptr: *mut vk::wl_display, surface_ptr: *mut vk::wl_surface) -> Result<Self, String> {
        unsafe {
            let entry = Entry::load().map_err(|e| e.to_string())?;
            let instance = create_instance(&entry)?;

            let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);
            let wayland_surface_fn =
                ash::khr::wayland_surface::Instance::new(&entry, &instance);

            let surface_create_info = vk::WaylandSurfaceCreateInfoKHR::default()
                .display(display_ptr)
                .surface(surface_ptr);
            let surface = wayland_surface_fn
                .create_wayland_surface(&surface_create_info, None)
                .map_err(|e| format!("create wayland surface: {e}"))?;

            let (physical_device, queue_family) =
                pick_physical_device(&instance, &surface_fn, surface)?;

            // Find a queue family that supports both graphics and presentation.
            let device = create_device(&instance, physical_device, queue_family)?;
            let queue = device.get_device_queue(queue_family, 0);

            let swapchain = ash::khr::swapchain::Device::new(&instance, &device);

            Ok(Self {
                _entry: entry,
                instance,
                surface,
                surface_fn,
                physical_device,
                device,
                queue,
                queue_family,
                swapchain,
                swapchain_khr: vk::SwapchainKHR::null(),
                images: Vec::new(),
                image_views: Vec::new(),
                format: vk::Format::UNDEFINED,
                extent: vk::Extent2D::default(),
                command_pool: vk::CommandPool::null(),
                command_buffer: vk::CommandBuffer::null(),
                acquire_semaphore: vk::Semaphore::null(),
                render_semaphore: vk::Semaphore::null(),
            })
        }
    }

    /// (Re)create the swapchain at `width`x`height` and present a cleared frame.
    fn present(&mut self, width: u32, height: u32) -> Result<(), String> {
        unsafe {
            // Recreate if the requested size differs from the current swapchain.
            if self.swapchain_khr == vk::SwapchainKHR::null()
                || self.extent.width != width
                || self.extent.height != height
            {
                self.create_swapchain(width, height)?;
            }

            // Wait for the previous frame's GPU work to finish so the semaphores
            // are free to reuse. Rendering is event-driven and infrequent, so a
            // full idle wait is acceptable.
            self.device.queue_wait_idle(self.queue).unwrap();

            let (image_index, _suboptimal) = self
                .swapchain
                .acquire_next_image(
                    self.swapchain_khr,
                    u64::MAX,
                    self.acquire_semaphore,
                    vk::Fence::null(),
                )
                .map_err(|e| format!("acquire next image: {e}"))?;

            let image = self.images[image_index as usize];

            // Record a command buffer that clears the image to CLEAR_COLOR.
            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
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
                self.command_buffer,
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
                self.command_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_value,
                std::slice::from_ref(&clear_range),
            );

            // TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR
            let to_present = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(subresource)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_present),
            );

            self.device
                .end_command_buffer(self.command_buffer)
                .map_err(|e| format!("end command buffer: {e}"))?;

            // Wait for acquire, submit, signal render.
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(std::slice::from_ref(&self.acquire_semaphore))
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(std::slice::from_ref(&self.command_buffer))
                .signal_semaphores(std::slice::from_ref(&self.render_semaphore));
            self.device
                .queue_submit(
                    self.queue,
                    std::slice::from_ref(&submit),
                    vk::Fence::null(),
                )
                .map_err(|e| format!("queue submit: {e}"))?;

            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(std::slice::from_ref(&self.render_semaphore))
                .swapchains(std::slice::from_ref(&self.swapchain_khr))
                .image_indices(std::slice::from_ref(&image_index));
            let result = self.swapchain.queue_present(self.queue, &present);

            // Out of date only means the swapchain must be recreated next time.
            match result {
                Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
                | Err(vk::Result::SUBOPTIMAL_KHR) => Ok(()),
                Err(e) => Err(format!("queue present: {e}")),
            }
        }
    }

    fn create_swapchain(&mut self, width: u32, height: u32) -> Result<(), String> {
        unsafe {
            if self.swapchain_khr != vk::SwapchainKHR::null() {
                // Ensure the old swapchain (and its present image uses) have fully
                // settled before destroying it.
                self.device.device_wait_idle().unwrap();
                self.swapchain.destroy_swapchain(self.swapchain_khr, None);
                self.swapchain_khr = vk::SwapchainKHR::null();
            }

            let caps = self
                .surface_fn
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .map_err(|e| format!("surface capabilities: {e}"))?;

            let formats = self
                .surface_fn
                .get_physical_device_surface_formats(self.physical_device, self.surface)
                .map_err(|e| format!("surface formats: {e}"))?;

            // Prefer the most common colour format.
            let format = formats
                .iter()
                .find(|f| f.format == vk::Format::B8G8R8A8_UNORM && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .or_else(|| formats.first())
                .ok_or("no surface formats")?;
            let format = format.format;

            let present_modes = self
                .surface_fn
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
                .map_err(|e| format!("present modes: {e}"))?;
            let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else {
                vk::PresentModeKHR::FIFO
            };

            let extent = vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            };

            let mut image_count = caps.min_image_count + 1;
            if caps.max_image_count > 0 && image_count > caps.max_image_count {
                image_count = caps.max_image_count;
            }

            let create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(image_count)
                .image_format(format)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true);

            self.swapchain_khr = self
                .swapchain
                .create_swapchain(&create_info, None)
                .map_err(|e| format!("create swapchain: {e}"))?;
            self.format = format;
            self.extent = extent;
            self.images = self
                .swapchain
                .get_swapchain_images(self.swapchain_khr)
                .map_err(|e| format!("get swapchain images: {e}"))?;

            // Recreate image views.
            for view in self.image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            for &image in &self.images {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    );
                let view = self
                    .device
                    .create_image_view(&view_info, None)
                    .map_err(|e| format!("create image view: {e}"))?;
                self.image_views.push(view);
            }

            // One command buffer + semaphores for the (re)created swapchain.
            if self.command_pool == vk::CommandPool::null() {
                let pool_info = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
                self.command_pool = self
                    .device
                    .create_command_pool(&pool_info, None)
                    .map_err(|e| format!("create command pool: {e}"))?;

                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let bufs = self
                    .device
                    .allocate_command_buffers(&alloc_info)
                    .map_err(|e| format!("allocate command buffer: {e}"))?;
                self.command_buffer = bufs[0];
            }

            if self.acquire_semaphore == vk::Semaphore::null() {
                let sem_info = vk::SemaphoreCreateInfo::default();
                self.acquire_semaphore = self
                    .device
                    .create_semaphore(&sem_info, None)
                    .map_err(|e| format!("create acquire semaphore: {e}"))?;
                self.render_semaphore = self
                    .device
                    .create_semaphore(&sem_info, None)
                    .map_err(|e| format!("create render semaphore: {e}"))?;
            }

            Ok(())
        }
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            if self.swapchain_khr != vk::SwapchainKHR::null() {
                self.swapchain.destroy_swapchain(self.swapchain_khr, None);
            }
            for &view in &self.image_views {
                self.device.destroy_image_view(view, None);
            }
            if self.acquire_semaphore != vk::Semaphore::null() {
                self.device.destroy_semaphore(self.acquire_semaphore, None);
                self.device.destroy_semaphore(self.render_semaphore, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            self.device.destroy_device(None);
            self.surface_fn.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_instance(entry: &Entry) -> Result<Instance, String> {
    unsafe {
        let app_name = c_str("wayvek");
        let engine_name = c_str("wayvek");
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::make_api_version(0, 1, 3, 0))
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&engine_name)
            .engine_version(1);

        let extension_names = [
            ash::khr::surface::NAME.as_ptr(),
            ash::khr::wayland_surface::NAME.as_ptr(),
        ];

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        entry
            .create_instance(&create_info, None)
            .map_err(|e| format!("create instance: {e}"))
    }
}

fn pick_physical_device(
    instance: &Instance,
    surface_fn: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32), String> {
    unsafe {
        let devices = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("enumerate physical devices: {e}"))?;

        for device in devices {
            let props = instance.get_physical_device_properties(device);
            let name = cstr_to_string(props.device_name.as_ptr());
            let queue_families = instance.get_physical_device_queue_family_properties(device);
            for (i, qf) in queue_families.iter().enumerate() {
                let supports_graphics = qf.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let supports_present = surface_fn
                    .get_physical_device_surface_support(device, i as u32, surface)
                    .unwrap_or(false);
                if supports_graphics && supports_present {
                    eprintln!("using GPU: {name}");
                    return Ok((device, i as u32));
                }
            }
        }
        Err("no suitable physical device".into())
    }
}

fn create_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
) -> Result<ash::Device, String> {
    unsafe {
        let priority = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priority);

        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions);

        instance
            .create_device(physical_device, &create_info, None)
            .map_err(|e| format!("create device: {e}"))
    }
}

fn c_str(s: &str) -> CString {
    CString::new(s).unwrap()
}

unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe {
        let mut len = 0;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let bytes = std::slice::from_raw_parts(ptr as *const u8, len);
        String::from_utf8_lossy(bytes).into_owned()
    }
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
        if let wl_registry::Event::Global { name, interface, .. } = event {
            match interface.as_str() {
                "wl_compositor" => {
                    let proxy =
                        registry.bind::<wl_compositor::WlCompositor, _, _>(name, 1, qh, AppData);
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
    let backend: Backend = conn.backend();
    let display_ptr = backend.display_ptr() as *mut vk::wl_display;

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
        size: (WINDOW_WIDTH, WINDOW_HEIGHT),
        vk: None,
        _surface_ptr: None,
        _display_ptr: None,
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
    state.init_window(&qh, display_ptr);

    // Blocking dispatch loop until the window is closed by the compositor.
    while state.running {
        event_queue
            .blocking_dispatch(&mut state)
            .expect("error dispatching Wayland events");
    }
}

// Keep ObjectId in scope for potential future use.
#[allow(dead_code)]
fn _assert_objectid(_: ObjectId) {}
