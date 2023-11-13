// Based on: https://github.com/gfx-rs/wgpu/blob/trunk/examples/mipmap/src/main.rs

static SHADER_SOURCE: &str = r#"
struct MipMapVertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn mipmap_gen_vertex_shader_main(@builtin(vertex_index) vertex_index: u32) -> MipMapVertexOutput {
  var out: MipMapVertexOutput;
  let x = i32(vertex_index) / 2;
  let y = i32(vertex_index) & 1;
  let tc = vec2<f32>(
    f32(x) * 2.0,
    f32(y) * 2.0,
  );
  out.clip_position = vec4<f32>(
    tc.x * 2.0 - 1.0,
    1.0 - tc.y * 2.0,
    0.0, 1.0,
  );
  out.uv = tc;
  return out;
}

@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;

@fragment
fn mipmap_gen_fragment_shader_main(in: MipMapVertexOutput) -> @location(0) vec4<f32> {
  return textureSample(t_diffuse, s_diffuse, in.uv);
}
"#;

pub struct MipMapGen {
  pipeline: wgpu::RenderPipeline,
  sampler:  wgpu::Sampler,
}

impl MipMapGen {
  pub fn new(device: &wgpu::Device, texture_format: wgpu::TextureFormat) -> Self {
    let mut target: wgpu::ColorTargetState = texture_format.into();
    target.blend = Some(wgpu::BlendState::ALPHA_BLENDING);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label:  Some("mipmap_gen_shader_module"),
      source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SOURCE)),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      entries: &[
        wgpu::BindGroupLayoutEntry {
          binding:    0,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty:         wgpu::BindingType::Texture {
            multisampled:   false,
            view_dimension: wgpu::TextureViewDimension::D2,
            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
          },
          count:      None,
        },
        wgpu::BindGroupLayoutEntry {
          binding:    1,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
          count:      None,
        },
      ],
      label:   Some("mipmap_gen_bind_group_layout"),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label:                Some("mipmap_gen_pipeline_layout"),
      bind_group_layouts:   &[&bind_group_layout],
      push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label:         Some("mipmap_gen_pipeline"),
      layout:        Some(&pipeline_layout),
      vertex:        wgpu::VertexState {
        module:      &shader,
        entry_point: "mipmap_gen_vertex_shader_main",
        buffers:     &[],
      },
      fragment:      Some(wgpu::FragmentState {
        module:      &shader,
        entry_point: "mipmap_gen_fragment_shader_main",
        targets:     &[Some(target)],
      }),
      primitive:     wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        polygon_mode: wgpu::PolygonMode::Fill,
        ..Default::default()
      },
      depth_stencil: None,
      multisample:   wgpu::MultisampleState::default(),
      multiview:     None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
      label: Some("mipmap_gen_sampler"),
      address_mode_u: wgpu::AddressMode::ClampToEdge,
      address_mode_v: wgpu::AddressMode::ClampToEdge,
      address_mode_w: wgpu::AddressMode::ClampToEdge,
      mag_filter: wgpu::FilterMode::Linear,
      min_filter: wgpu::FilterMode::Linear,
      mipmap_filter: wgpu::FilterMode::Nearest,
      ..Default::default()
    });
    Self { pipeline, sampler }
  }

  pub fn generate_mipmaps(
    &self,
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    texture: &wgpu::Texture,
    mip_count: u32,
  ) {
    let bind_group_layout = self.pipeline.get_bind_group_layout(0);
    for target_mip in 1..mip_count {
      let create_view = |base_mip_level| {
        texture.create_view(&wgpu::TextureViewDescriptor {
          label: Some("mipmap_read"),
          format: None,
          dimension: None,
          aspect: wgpu::TextureAspect::All,
          base_mip_level,
          mip_level_count: Some(1),
          base_array_layer: 0,
          array_layer_count: None,
        })
      };
      let read_view = create_view(target_mip - 1);
      let write_view = create_view(target_mip);
      let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout:  &bind_group_layout,
        entries: &[
          wgpu::BindGroupEntry {
            binding:  0,
            resource: wgpu::BindingResource::TextureView(&read_view),
          },
          wgpu::BindGroupEntry {
            binding:  1,
            resource: wgpu::BindingResource::Sampler(&self.sampler),
          },
        ],
        label:   None,
      });

      let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label:                    None,
        color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
          view:           &write_view,
          resolve_target: None,
          ops:            wgpu::Operations {
            load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: true,
          },
        })],
        depth_stencil_attachment: None,
      });
      rpass.set_pipeline(&self.pipeline);
      rpass.set_bind_group(0, &bind_group, &[]);
      rpass.draw(0..3, 0..1);
    }
  }
}
