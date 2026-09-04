use wayland_client::Connection;

use wayvek::backend::wayland::WaylandState;
use wayvek::negotiate;
use wayvek::renderer::vulkan::{Vulkan, drm_fourcc_to_vk};
use wayvek::swapchain::Swapchain;

fn main() {
    let conn = Connection::connect_to_env().expect("failed to connect to Wayland display");

    let mut event_queue = conn.new_event_queue();
    let qh = event_queue.handle();

    let mut wayland = WaylandState::new();

    // First roundtrip: discover globals and queue bind requests (including
    // zwp_linux_dmabuf_v1). Second roundtrip: flush those binds and process
    // the feedback events (MainDevice, FormatTable, TrancheFormats).
    {
        let display = conn.display();
        let _registry = display.get_registry(&qh, wayvek::backend::wayland::GlobalData);
        event_queue
            .roundtrip(&mut wayland)
            .expect("failed to roundtrip while reading globals");
        event_queue
            .roundtrip(&mut wayland)
            .expect("failed to roundtrip while reading dmabuf feedback");
    }

    let main_device = wayland.main_device().expect("no main DRM device");
    let advertised = wayland.advertised_formats();

    // Create the Vulkan renderer on the matching GPU.
    let mut renderer = Vulkan::new(main_device).expect("vulkan init failed");

    // Negotiate a mutually supported format and modifier.
    let (fourcc, modifier) =
        negotiate(&advertised, |f| renderer.get_supported_modifiers(f))
            .expect("no mutually supported DRM format");
    let _vk_format = drm_fourcc_to_vk(drm_fourcc::DrmFourcc::try_from(fourcc).unwrap());
    renderer.set_buffer_format(fourcc, modifier);

    eprintln!("chose format={fourcc} modifier={modifier:016x}");

    // Create the swapchain (manages DRM syncobj + double-buffer coordination).
    let mut swapchain = Swapchain::new(main_device).expect("swapchain init failed");

    // Import the syncobj timeline into the compositor.
    let timeline_fd = swapchain.timeline_fd().expect("failed to export timeline fd");
    wayland.import_timeline(&timeline_fd, &qh);

    // Create the window now that all globals are known.
    wayland.init_window(&qh);

    // Blocking dispatch loop until the window is closed by the compositor.
    while wayland.running {
        event_queue
            .blocking_dispatch(&mut wayland)
            .expect("error dispatching Wayland events");

        if let Some((w, h)) = wayland.pending_render.take() {
            match swapchain.next_frame(&mut renderer, w, h) {
                Ok((frame, sync)) => wayland.commit(frame, sync),
                Err(e) => eprintln!("frame error: {e}"),
            }
        }
    }
}
