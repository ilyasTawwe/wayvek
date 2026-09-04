use wayland_client::{
    Connection, Dispatch, QueueHandle,
    protocol::{wl_buffer, wl_compositor, wl_registry, wl_surface},
};
use wayland_protocols::xdg::decoration::zv1::client::{
    zxdg_decoration_manager_v1, zxdg_toplevel_decoration_v1,
};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};
use wayland_protocols::wp::single_pixel_buffer::v1::client::wp_single_pixel_buffer_manager_v1;
use wayland_protocols::wp::viewporter::client::{wp_viewport, wp_viewporter};

const WINDOW_WIDTH: i32 = 800;
const WINDOW_HEIGHT: i32 = 600;

// Helper to convert an 8-bit color channel (0-255) to the 32-bit percentage
// value expected by wp_single_pixel_buffer_manager_v1 (0 - UINT32_MAX).
// 255 * 0x01010101 == 0xffffffff.
const fn percent(c: u8) -> u32 {
    (c as u32) * 0x0101_0101
}

// Premultiplied RGBA, opaque indigo (#2e2ea3).
const COLOR_R: u32 = percent(0x2e);
const COLOR_G: u32 = percent(0x2e);
const COLOR_B: u32 = percent(0xa3);
const COLOR_A: u32 = u32::MAX;

struct State {
    running: bool,
    compositor: Option<wl_compositor::WlCompositor>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    viewporter: Option<wp_viewporter::WpViewporter>,
    single_pixel: Option<wp_single_pixel_buffer_manager_v1::WpSinglePixelBufferManagerV1>,
    decoration_manager: Option<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1>,
    surface: Option<wl_surface::WlSurface>,
    viewport: Option<wp_viewport::WpViewport>,
    xdg_surface: Option<xdg_surface::XdgSurface>,
    toplevel: Option<xdg_toplevel::XdgToplevel>,
    decoration: Option<zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1>,
    buffer: Option<wl_buffer::WlBuffer>,
    size: (u32, u32),
}

impl State {
    /// Create the surface, viewport and xdg toplevel once all needed globals
    /// have been bound.
    fn init_window(&mut self, qh: &QueueHandle<Self>) {
        if self.surface.is_some() {
            return;
        }
        let (Some(compositor), Some(wm_base), Some(viewporter)) =
            (&self.compositor, &self.wm_base, &self.viewporter)
        else {
            return;
        };

        let surface = compositor.create_surface(qh, AppData);
        let viewport = viewporter.get_viewport(&surface, qh, AppData);
        let xdg_surface = wm_base.get_xdg_surface(&surface, qh, AppData);
        let toplevel = xdg_surface.get_toplevel(qh, AppData);
        toplevel.set_title(String::from("wayvek"));
        toplevel.set_app_id(String::from("wayvek"));
        toplevel.set_min_size(1, 1);

        self.surface = Some(surface);
        self.viewport = Some(viewport);
        self.xdg_surface = Some(xdg_surface);
        self.toplevel = Some(toplevel);

        // Request server-side window decorations if the compositor offers them.
        if let Some(decoration_manager) = &self.decoration_manager {
            let decoration = decoration_manager.get_toplevel_decoration(
                self.toplevel.as_ref().unwrap(),
                qh,
                AppData,
            );
            decoration.set_mode(zxdg_toplevel_decoration_v1::Mode::ServerSide);
            self.decoration = Some(decoration);
        }

        // Ask the compositor to configure (and map) our window.
        self.surface.as_ref().unwrap().commit();
    }

    /// Attach the (single) single-pixel buffer and scale it to the given size
    /// via the viewporter, then commit. The buffer is created once and reused,
    /// since single-pixel buffers can be re-attached after the compositor
    /// releases them.
    fn redraw(&mut self, qh: &QueueHandle<Self>, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        let (Some(surface), Some(viewport), Some(single_pixel)) =
            (&self.surface, &self.viewport, &self.single_pixel)
        else {
            return;
        };

        if self.buffer.is_none() {
            self.buffer = Some(
                single_pixel.create_u32_rgba_buffer(COLOR_R, COLOR_G, COLOR_B, COLOR_A, qh, AppData),
            );
        }

        surface.attach(self.buffer.as_ref(), 0, 0);
        viewport.set_destination(width as i32, height as i32);
        surface.commit();
    }
}

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
                "wp_viewporter" => {
                    let proxy =
                        registry.bind::<wp_viewporter::WpViewporter, _, _>(name, 1, qh, AppData);
                    state.viewporter = Some(proxy);
                }
                "wp_single_pixel_buffer_manager_v1" => {
                    let proxy = registry.bind::<
                        wp_single_pixel_buffer_manager_v1::WpSinglePixelBufferManagerV1,
                        _,
                        _,
                    >(name, 1, qh, AppData);
                    state.single_pixel = Some(proxy);
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
            state.init_window(qh);
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
        // Reply to ping so the compositor keeps us alive.
        if let xdg_wm_base::Event::Ping { serial } = event {
            proxy.pong(serial);
        }
    }
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

impl Dispatch<wp_viewporter::WpViewporter, AppData> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wp_viewporter::WpViewporter,
        _event: wp_viewporter::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wp_single_pixel_buffer_manager_v1::WpSinglePixelBufferManagerV1, AppData> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wp_single_pixel_buffer_manager_v1::WpSinglePixelBufferManagerV1,
        _event: wp_single_pixel_buffer_manager_v1::Event,
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

impl Dispatch<wp_viewport::WpViewport, AppData> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wp_viewport::WpViewport,
        _event: wp_viewport::Event,
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
        qh: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            xdg_surface.ack_configure(serial);
            // Commit our content on every configure so the compositor can map
            // and (re)size the window.
            let (w, h) = state.size;
            state.redraw(qh, w, h);
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
            // Remember the configured size so we can render it.
            xdg_toplevel::Event::Configure { width, height, .. } => {
                if width != 0 && height != 0 {
                    state.size = (width as u32, height as u32);
                }
            }
            // If the compositor closes our window, exit.
            xdg_toplevel::Event::Close => {
                state.running = false;
            }
            _ => {}
        }
    }
}

fn main() {
    let conn = Connection::connect_to_env().expect("failed to connect to Wayland display");

    let mut event_queue = conn.new_event_queue();
    let qh = event_queue.handle();

    let mut state = State {
        running: true,
        compositor: None,
        wm_base: None,
        viewporter: None,
        single_pixel: None,
        decoration_manager: None,
        surface: None,
        viewport: None,
        xdg_surface: None,
        toplevel: None,
        decoration: None,
        buffer: None,
        size: (WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32),
    };

    // Roundtrip once to read the globals and create the window.
    {
        let display = conn.display();
        let _registry = display.get_registry(&qh, GlobalData);
        event_queue
            .roundtrip(&mut state)
            .expect("failed to roundtrip while reading globals");
    }

    eprintln!(
        "wayvek: window created; requires a compositor supporting wp_single_pixel_buffer_manager_v1 and wp_viewporter"
    );

    // Blocking dispatch loop until the window is closed by the compositor.
    while state.running {
        event_queue
            .blocking_dispatch(&mut state)
            .expect("error dispatching Wayland events");
    }
}
