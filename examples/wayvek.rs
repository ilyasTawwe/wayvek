use anyhow::{anyhow, Result};
use ash::vk;
use wayland_client::Connection;

use wayvek::backend::wayland::WaylandState;
use wayvek::drm::{DrmSyncobj, open_render_node};
use wayvek::negotiate;
use wayvek::renderer::vulkan::Vulkan;
use wayvek::swapchain::Swapchain;
use wayvek::ExplicitSync;

/// Window color as normalized floats (0.0-1.0) used as the render clear color.
const CLEAR_COLOR: [f32; 4] = [0.18, 0.18, 0.64, 1.0];

/// GPU resources for drawing the triangle. Built once; independent of window
/// size because viewport/scissor are dynamic.
struct TrianglePipeline {
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
}

impl Drop for TrianglePipeline {
    fn drop(&mut self) {
        // SAFETY: all handles were created on `self.device` and are still
        // valid; no other thread uses them at drop time.
        unsafe {
            self.device.destroy_shader_module(self.vert_module, None);
            self.device.destroy_shader_module(self.frag_module, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// Decode a SPIR-V byte blob into the `u32` words `vkCreateShaderModule` wants.
fn load_spirv(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect()
}

/// Build a graphics pipeline that renders a full-screen-style triangle with a
/// hardcoded vertex/color table (no vertex buffers or descriptor sets).
fn create_triangle_pipeline(device: &ash::Device, format: vk::Format) -> Result<TrianglePipeline> {
    // SAFETY: all Vulkan handles created here are valid for `device` and the
    // p_next chain is `#[repr(C)]`. Created handles are destroyed either by the
    // returned struct's `Drop` or, on the `create_graphics_pipelines` error
    // path, explicitly below.
    unsafe {
        let vert_code = load_spirv(include_bytes!("../shaders/triangle.vert.spv"));
        let frag_code = load_spirv(include_bytes!("../shaders/triangle.frag.spv"));

        let vert_module = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_code), None)?;
        let frag_module = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_code), None)?;
        let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)?;

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(c"main"),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            );
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&blend_attachment));
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Dynamic rendering (core in Vulkan 1.3): no render pass needed.
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(std::slice::from_ref(&format));

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(vk::RenderPass::null())
            .subpass(0)
            .push_next(&mut rendering);

        let pipelines = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(partial, e)| {
                for p in &partial {
                    device.destroy_pipeline(*p, None);
                }
                e
            })?;

        Ok(TrianglePipeline {
            device: device.clone(),
            layout,
            pipeline: pipelines[0],
            vert_module,
            frag_module,
        })
    }
}

fn main() -> Result<()> {
    let conn = Connection::connect_to_env()?;

    let mut event_queue = conn.new_event_queue();
    let qh = event_queue.handle();

    let mut wayland = WaylandState::new();

    // First roundtrip: discover globals and queue bind requests (including
    // zwp_linux_dmabuf_v1). Second roundtrip: flush those binds and process
    // the feedback events (MainDevice, FormatTable, TrancheFormats).
    {
        let display = conn.display();
        let _registry = display.get_registry(&qh, wayvek::backend::wayland::GlobalData);
        event_queue.roundtrip(&mut wayland)?;
        event_queue.roundtrip(&mut wayland)?;
    }

    let main_device = wayland.main_device().ok_or_else(|| anyhow!("no main DRM device"))?;
    let advertised = wayland.advertised_formats();

    // Create the Vulkan renderer on the matching GPU.
    let mut renderer = Vulkan::new(main_device)?;

    // Negotiate a mutually supported format and modifier.
    let (fourcc, modifier) = negotiate(&advertised, |f| renderer.get_supported_modifiers(f))?;
    renderer.set_buffer_format(fourcc, modifier);

    eprintln!("chose format={fourcc} modifier={modifier:016x}");

    // Create the DRM syncobj timeline on the render node. main owns this
    // object and does the sync bridging per frame below.
    let drm_sync = DrmSyncobj::create(open_render_node(main_device)?)?;

    // Import the syncobj timeline into the compositor once.
    wayland.import_timeline(&drm_sync.export_fd()?, &qh);

    // Create the swapchain (double-buffered buffer slot allocator).
    let mut swapchain = Swapchain::new();

    // Create the window now that all globals are known.
    wayland.init_window(&qh);

    // Build the triangle pipeline once (the buffer format is set above).
    let format = renderer.buffer_format().expect("buffer format configured");
    let triangle = create_triangle_pipeline(&renderer.device(), format)?;

    // Blocking dispatch loop until the window is closed by the compositor.
    while wayland.running {
        event_queue.blocking_dispatch(&mut wayland)?;

        if let Some((w, h)) = wayland.pending_render.take() {
            match (|| -> Result<()> {
                // 1. Acquire the next buffer slot (with backpressure).
                let slot = swapchain.acquire(&drm_sync)?;

                // 2. Record the frame into the buffer.
                let (frame, sync_file) =
                    renderer.draw(slot, w, h, |_slot, device, cmd, image, view, _format| {
                        let subresource = vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1);

                        // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
                        let to_attachment = vk::ImageMemoryBarrier::default()
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(image)
                            .subresource_range(subresource)
                            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
                        unsafe {
                            device.cmd_pipeline_barrier(
                                cmd,
                                vk::PipelineStageFlags::TOP_OF_PIPE,
                                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                                vk::DependencyFlags::empty(),
                                &[],
                                &[],
                                std::slice::from_ref(&to_attachment),
                            );
                        }

                        let clear_value = vk::ClearValue {
                            color: vk::ClearColorValue { float32: CLEAR_COLOR },
                        };
                        let color_attachment = vk::RenderingAttachmentInfo::default()
                            .image_view(view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(clear_value);
                        let rendering = vk::RenderingInfo::default()
                            .render_area(vk::Rect2D {
                                offset: vk::Offset2D::default(),
                                extent: vk::Extent2D { width: w, height: h },
                            })
                            .layer_count(1)
                            .color_attachments(std::slice::from_ref(&color_attachment));
                        unsafe {
                            device.cmd_begin_rendering(cmd, &rendering);
                        }

                        let viewport = vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: w as f32,
                            height: h as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        };
                        let scissor = vk::Rect2D {
                            offset: vk::Offset2D::default(),
                            extent: vk::Extent2D { width: w, height: h },
                        };
                        unsafe {
                            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport));
                            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
                            device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                triangle.pipeline,
                            );
                            device.cmd_draw(cmd, 3, 1, 0, 0);
                            device.cmd_end_rendering(cmd);
                        }

                        // COLOR_ATTACHMENT_OPTIMAL -> GENERAL (compositor read)
                        let to_general = vk::ImageMemoryBarrier::default()
                            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::GENERAL)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(image)
                            .subresource_range(subresource)
                            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
                        unsafe {
                            device.cmd_pipeline_barrier(
                                cmd,
                                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                                vk::DependencyFlags::empty(),
                                &[],
                                &[],
                                std::slice::from_ref(&to_general),
                            );
                        }
                    })?;

                // 3. Sync bridging: import the completion sync-file into the
                //    timeline, producing the acquire/release points.
                let acquire_point = drm_sync.import_sync_file(&sync_file)?;
                let release_point = acquire_point + 1;
                swapchain.mark_presented(slot, release_point);

                // 4. Hand the frame + sync points to the Wayland backend.
                wayland.commit(
                    frame,
                    ExplicitSync {
                        acquire_point,
                        release_point,
                    },
                );
                Ok(())
            })() {
                Ok(()) => {}
                Err(e) => eprintln!("frame error: {e:#}"),
            }
        }
    }

    Ok(())
}