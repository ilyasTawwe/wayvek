pub mod dmabuf;

use std::os::unix::io::{AsFd, OwnedFd};

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

use zerocopy::FromBytes;

use crate::{DmaFrame, ExplicitSync};
use dmabuf::{DrmFormatTable, FormatInfo};

struct AppData;
pub struct GlobalData;

pub struct WaylandState {
    pub running: bool,
    compositor: Option<wl_compositor::WlCompositor>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    decoration_manager: Option<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1>,
    surface: Option<wl_surface::WlSurface>,
    xdg_surface: Option<xdg_surface::XdgSurface>,
    toplevel: Option<xdg_toplevel::XdgToplevel>,
    decoration: Option<zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1>,
    dmabuf: Option<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1>,
    feedback: Option<zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1>,
    main_device: Option<u64>,
    format_table: Option<DrmFormatTable>,
    dmabuf_formats: Vec<FormatInfo>,
    pub size: (u32, u32),
    /// Set by xdg_surface::Configure; consumed by the main loop to trigger rendering.
    pub pending_render: Option<(u32, u32)>,
    syncobj_manager: Option<WpLinuxDrmSyncobjManagerV1>,
    surface_sync: Option<WpLinuxDrmSyncobjSurfaceV1>,
    syncobj_timeline: Option<WpLinuxDrmSyncobjTimelineV1>,
    pending_acquire: Option<u64>,
    pending_release: Option<u64>,
    qh: Option<QueueHandle<Self>>,
    create_size: Option<(i32, i32)>,
}

impl Default for WaylandState {
    fn default() -> Self {
        Self::new()
    }
}

impl WaylandState {
    pub fn new() -> Self {
        Self {
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
            size: (800, 600),
            pending_render: None,
            syncobj_manager: None,
            surface_sync: None,
            syncobj_timeline: None,
            pending_acquire: None,
            pending_release: None,
            qh: None,
            create_size: None,
        }
    }

    /// Create the surface + xdg toplevel once all needed globals are bound.
    pub fn init_window(&mut self, qh: &QueueHandle<Self>) {
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

        let decoration = self.decoration_manager.as_ref().map(|m| {
            let deco = m.get_toplevel_decoration(&toplevel, qh, AppData);
            deco.set_mode(zxdg_toplevel_decoration_v1::Mode::ServerSide);
            deco
        });

        self.surface = Some(surface.clone());
        self.xdg_surface = Some(xdg_surface);
        self.toplevel = Some(toplevel);
        self.decoration = decoration;

        let (Some(manager), Some(surf)) = (&self.syncobj_manager, self.surface.as_ref()) else {
            panic!("wp_linux_drm_syncobj_manager_v1 is not available");
        };
        let surface_sync = manager.get_surface(surf, qh, AppData);
        self.surface_sync = Some(surface_sync);

        self.qh = Some(qh.clone());
        surface.commit();
    }

    pub fn main_device(&self) -> Option<u64> {
        self.main_device
    }

    pub fn advertised_formats(&self) -> Vec<(u32, Vec<u64>)> {
        self.dmabuf_formats
            .iter()
            .map(|f| (f.code as u32, f.modifiers.clone()))
            .collect()
    }

    /// Import the DRM syncobj timeline fd into the compositor once.
    pub fn import_timeline(&mut self, fd: &OwnedFd, qh: &QueueHandle<Self>) {
        if self.syncobj_timeline.is_some() {
            return;
        }
        let Some(manager) = &self.syncobj_manager else {
            return;
        };
        let timeline = manager.import_timeline(fd.as_fd(), qh, AppData);
        self.syncobj_timeline = Some(timeline);
    }

    /// Wrap the DMA-BUF in a wl_buffer and commit it to the compositor with
    /// explicit synchronization points.
    pub fn commit(&mut self, frame: DmaFrame, sync: ExplicitSync) {
        let (Some(dmabuf), Some(qh)) = (&self.dmabuf.clone(), self.qh.clone()) else {
            return;
        };

        self.pending_acquire = Some(sync.acquire_point);
        self.pending_release = Some(sync.release_point);

        let params = dmabuf.create_params(&qh, AppData);
        for (idx, plane) in frame.planes.iter().enumerate() {
            params.add(
                frame.fd.as_fd(),
                idx as u32,
                plane.offset as u32,
                plane.stride as u32,
                (frame.modifier >> 32) as u32,
                (frame.modifier & 0xffff_ffff) as u32,
            );
        }
        params.create(
            frame.width as i32,
            frame.height as i32,
            frame.format,
            zwp_linux_buffer_params_v1::Flags::empty(),
        );

        self.create_size = Some((frame.width as i32, frame.height as i32));
    }

    fn add_wayland_modifier(&mut self, code: DrmFourcc, modifier: u64) {
        if let Some(fmt) = self.dmabuf_formats.iter_mut().find(|f| f.code == code) {
            if !fmt.modifiers.contains(&modifier) {
                fmt.modifiers.push(modifier);
            }
        } else {
            self.dmabuf_formats.push(FormatInfo {
                code,
                modifiers: vec![modifier],
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch implementations
// ---------------------------------------------------------------------------

impl Dispatch<wl_registry::WlRegistry, GlobalData> for WaylandState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _data: &GlobalData,
        _conn: &Connection,
        qh: &QueueHandle<WaylandState>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
            ..
        } = event
        {
            match interface.as_str() {
                "wl_compositor" => {
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
                "zwp_linux_dmabuf_v1" => {
                    let proxy =
                        registry.bind::<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1, _, _>(
                            name, 5, qh, AppData,
                        );
                    let feedback = proxy.get_default_feedback(qh, AppData);
                    state.dmabuf = Some(proxy);
                    state.feedback = Some(feedback);
                }
                "wp_linux_drm_syncobj_manager_v1" => {
                    let proxy = registry.bind::<WpLinuxDrmSyncobjManagerV1, _, _>(
                        name, 1, qh, AppData,
                    );
                    state.syncobj_manager = Some(proxy);
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, AppData> for WaylandState {
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

impl Dispatch<wl_compositor::WlCompositor, AppData> for WaylandState {
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

impl Dispatch<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1, AppData> for WaylandState {
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

impl Dispatch<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1, AppData> for WaylandState {
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

impl Dispatch<zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1, AppData> for WaylandState {
    fn event(
        state: &mut Self,
        _proxy: &zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1,
        event: zwp_linux_dmabuf_feedback_v1::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<WaylandState>,
    ) {
        match event {
            zwp_linux_dmabuf_feedback_v1::Event::MainDevice { device } => {
                let mut bytes = [0u8; 8];
                let n = device.len().min(8);
                bytes[..n].copy_from_slice(&device[..n]);
                state.main_device = Some(u64::from_le_bytes(bytes));
            }
            zwp_linux_dmabuf_feedback_v1::Event::FormatTable { fd, size } => {
                state.format_table = DrmFormatTable::map(fd, size);
            }
            zwp_linux_dmabuf_feedback_v1::Event::TrancheFormats { indices } => {
                let mut pairs: Vec<(DrmFourcc, u64)> = Vec::new();
                if let Some(table) = &state.format_table {
                    for pair in indices.as_chunks::<2>().0 {
                        let Ok(idx) = u16::ref_from_bytes(pair) else {
                            continue;
                        };
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

impl Dispatch<zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1, AppData> for WaylandState {
    fn event(
        state: &mut Self,
        params: &zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1,
        event: zwp_linux_buffer_params_v1::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            zwp_linux_buffer_params_v1::Event::Created { buffer } => {
                params.destroy();
                if let Some(surface) = &state.surface {
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
                    surface_sync.set_acquire_point(
                        timeline,
                        (acquire >> 32) as u32,
                        acquire as u32,
                    );
                    surface_sync.set_release_point(
                        timeline,
                        (release >> 32) as u32,
                        release as u32,
                    );

                    surface.attach(Some(&buffer), 0, 0);
                    let (w, h) = state.create_size.unwrap_or((0, 0));
                    surface.damage_buffer(0, 0, w, h);
                    surface.commit();
                }
                state.create_size = None;
            }
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
        WaylandState,
        zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1,
        [zwp_linux_buffer_params_v1::EVT_CREATED_OPCODE => (wl_buffer::WlBuffer, AppData)]
    );
}

impl Dispatch<wl_buffer::WlBuffer, AppData> for WaylandState {
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

impl Dispatch<WpLinuxDrmSyncobjManagerV1, AppData> for WaylandState {
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

impl Dispatch<WpLinuxDrmSyncobjSurfaceV1, AppData> for WaylandState {
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

impl Dispatch<WpLinuxDrmSyncobjTimelineV1, AppData> for WaylandState {
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

impl Dispatch<zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1, AppData> for WaylandState {
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

impl Dispatch<wl_surface::WlSurface, AppData> for WaylandState {
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

impl Dispatch<xdg_surface::XdgSurface, AppData> for WaylandState {
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
            state.pending_render = Some((w, h));
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, AppData> for WaylandState {
    fn event(
        state: &mut Self,
        _proxy: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _data: &AppData,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            xdg_toplevel::Event::Configure {
                width, height, ..
            } => {
                if width != 0 && height != 0 {
                    state.size = (width as u32, height as u32);
                }
            }
            xdg_toplevel::Event::Close => state.running = false,
            _ => {}
        }
    }
}
