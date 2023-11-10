use std::sync::Arc;
use std::sync::Mutex;

use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3};
use wgpu::util::DeviceExt;
use wgpu::BufferSlice;

pub struct Transform {
  pub translation: Vector3<f32>,
  pub rotation:    UnitQuaternion<f32>,
  pub scale:       f32,
}

impl Default for Transform {
  fn default() -> Self {
    Self {
      translation: Vector3::new(0.0, 0.0, 0.0),
      rotation:    UnitQuaternion::identity(),
      scale:       1.0,
    }
  }
}

impl Transform {
  pub fn apply(&self, point: Point3<f32>) -> Point3<f32> {
    return self.scale * (self.rotation * point) + self.translation;
  }
}

pub struct Projection {
  pub clip_near:         f32,
  pub clip_far:          f32,
  pub model_matrix:      Matrix4<f32>,
  pub view_matrix:       Matrix4<f32>,
  pub projection_matrix: Matrix4<f32>,
  pub derived_pvm:       Matrix4<f32>,
  pub derived_vm:        Matrix4<f32>,
}

impl Default for Projection {
  fn default() -> Self {
    Self {
      clip_near:         0.1,
      clip_far:          100.0,
      model_matrix:      Matrix4::identity(),
      view_matrix:       Matrix4::identity(),
      projection_matrix: Matrix4::identity(),
      derived_vm:        Matrix4::identity(),
      derived_pvm:       Matrix4::identity(),
    }
  }
}

impl Projection {
  fn recompute_derived(&mut self) {
    self.derived_vm = self.view_matrix * self.model_matrix;
    self.derived_pvm = self.projection_matrix * self.derived_vm;
  }

  pub fn set_model_matrix(&mut self, model_matrix: Matrix4<f32>) {
    self.model_matrix = model_matrix;
    self.recompute_derived();
  }

  pub fn set_view_matrix(&mut self, view_matrix: Matrix4<f32>) {
    self.view_matrix = view_matrix;
    self.recompute_derived();
  }

  pub fn set_projection_matrix(&mut self, projection_matrix: Matrix4<f32>) {
    self.projection_matrix = projection_matrix;
    self.recompute_derived();
  }

  pub fn set_model_transform(&mut self, model_transform: Transform) {
    let mut model_matrix = Matrix4::identity();
    let rotation = model_transform.rotation.to_rotation_matrix();
    model_matrix.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation.matrix());
    model_matrix.fixed_view_mut::<3, 1>(0, 3).copy_from(&model_transform.translation);
    model_matrix.m43 *= model_transform.scale;
    self.set_model_matrix(model_matrix);
  }

  #[rustfmt::skip]
  pub fn set_view_look_at(&mut self, eye: Point3<f32>, look_at: Point3<f32>) {
    let view = (look_at - eye).normalize();
    let right = Vector3::y().cross(&view).normalize();
    let up = view.cross(&right);
    #[rustfmt::skip]
    let view_matrix = Matrix4::new(
      right.x, right.y, right.z, -right.dot(&eye.coords),
      up.x,    up.y,    up.z,    -up.dot(&eye.coords),
      view.x,  view.y,  view.z,  -view.dot(&eye.coords),
      0.0,     0.0,     0.0,      1.0,
    );
    self.set_view_matrix(view_matrix);
  }

  /// fov_deg is the vertical field of view in degrees
  /// aspect is width / height
  #[rustfmt::skip]
  pub fn set_perspective_projection(&mut self, fov_deg: f32, aspect: f32, near: f32, far: f32) {
    self.clip_near = near;
    self.clip_far = far;
    let fov_rad = fov_deg * std::f32::consts::PI / 180.0;
    let height = 1.0 / (fov_rad / 2.0).tan();
    let width = height * aspect;
    let m22 = far / (far - near);
    #[rustfmt::skip]
    let projection_matrix = Matrix4::new(
      width, 0.0,    0.0,  0.0,
      0.0,   height, 0.0,  0.0,
      0.0,   0.0,    m22, -near * m22,
      0.0,   0.0,    1.0,  0.0,
    );
    self.set_projection_matrix(projection_matrix);
  }

  #[rustfmt::skip]
  pub fn set_ortho_projection(
    &mut self,
    w: f32,
    h: f32,
    near: f32,
    far: f32,
  ) {
    self.clip_near = near;
    self.clip_far = far;
    let n = near;
    let f = far;
    #[rustfmt::skip]
    let projection_matrix = Matrix4::new(
      2.0/w,  0.0,    0.0,      -1.0,
      0.0,   -2.0/h,  0.0,       1.0,
      0.0,    0.0,    1.0/(f-n), n/(f-n),
      0.0,    0.0,    0.0,       1.0,
    );
    self.set_projection_matrix(projection_matrix);
  }

  pub fn update_uniforms(&mut self, uniforms: &mut Uniforms) {
    uniforms.transform_m = self.model_matrix.into();
    uniforms.transform_vm = self.derived_vm.into();
    uniforms.transform_pvm = self.derived_pvm.into();
    if let Some(inv) = self.derived_pvm.try_inverse() {
      uniforms.transform_pvm_inv = inv.into();
    }
    uniforms.near = self.clip_near;
    uniforms.far = self.clip_far;
  }

  pub fn screen_space_to_world_space(&self, screen_space: Point3<f32>) -> Point3<f32> {
    let inv_pvm = self.derived_pvm.try_inverse().unwrap();
    inv_pvm.transform_point(&screen_space)
  }
}

// ===== wgpu interfacing =====

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Vertex {
  pub position: [f32; 3],
  pub color:    [f32; 3],
  pub uv:       [f32; 2],
}

impl Vertex {
  const ATTRIBS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

  pub fn desc() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
      array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
      step_mode:    wgpu::VertexStepMode::Vertex,
      attributes:   &Self::ATTRIBS,
    }
  }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default)]
pub struct VoxelInstance {
  pub voxel_position: [f32; 3],
  pub voxel_tint:     [f32; 3],
}

impl VoxelInstance {
  const ATTRIBS: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![3 => Float32x3, 4 => Float32x3];

  pub fn desc() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
      array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
      step_mode:    wgpu::VertexStepMode::Instance,
      attributes:   &Self::ATTRIBS,
    }
  }
}

// #[repr(C, align(16))]
// #[derive(Debug, Clone, Copy, Default)]
// pub struct RayPathVertex {
//   pub pos: [f32; 3],
// }

// impl RayPathVertex {
//   const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

//   pub fn desc() -> wgpu::VertexBufferLayout<'static> {
//     wgpu::VertexBufferLayout {
//       array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
//       step_mode:    wgpu::VertexStepMode::Instance,
//       attributes:   &Self::ATTRIBS,
//     }
//   }
// }

#[repr(C, align(16))]
#[derive(Debug, Clone)]
pub struct Uniforms {
  pub transform_m:       [[f32; 4]; 4],
  pub transform_vm:      [[f32; 4]; 4],
  pub transform_pvm:     [[f32; 4]; 4],
  pub transform_pvm_inv: [[f32; 4]; 4],
  pub near:              f32,
  pub far:               f32,
}

impl Uniforms {
  pub fn new() -> Self {
    Self {
      transform_m:       Matrix4::identity().into(),
      transform_vm:      Matrix4::identity().into(),
      transform_pvm:     Matrix4::identity().into(),
      transform_pvm_inv: Matrix4::identity().into(),
      near:              0.1,
      far:               1000.0,
    }
  }
}

pub unsafe fn ref_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
  std::slice::from_raw_parts((p as *const T) as *const u8, std::mem::size_of::<T>())
}

pub unsafe fn slice_as_u8_slice<T: Sized>(p: &[T]) -> &[u8] {
  std::slice::from_raw_parts((&p[0] as *const T) as *const u8, std::mem::size_of::<T>() * p.len())
}

pub struct ResizingBuffer<T> {
  pub label:          String,
  pub usage:          wgpu::BufferUsages,
  /// The size of the buffer in bytes is `std::mem::size_of::<T>() * len`.
  pub buffer_and_len: Option<(wgpu::Buffer, usize)>,
  pub contents:       Vec<T>,
}

impl<T> ResizingBuffer<T> {
  pub fn new(label: String, usage: wgpu::BufferUsages) -> Self {
    Self {
      label,
      usage,
      buffer_and_len: None,
      contents: Vec::new(),
    }
  }

  pub fn update<'a>(&'a mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> usize {
    let reallocate_length = match self.buffer_and_len.as_mut() {
      Some((_, buffer_len)) if self.contents.len() <= *buffer_len => None,
      Some((_, ref buffer_len)) => Some(self.contents.len().max(buffer_len + buffer_len / 2)),
      None => Some(self.contents.len().max(1)),
    };
    match reallocate_length {
      Some(reallocate_length) => {
        match self.buffer_and_len.take() {
          Some((buffer, _)) => buffer.destroy(),
          None => {}
        }
        self.buffer_and_len = Some((
          device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some(&self.label),
            size:               (std::mem::size_of::<T>() * reallocate_length)
              as wgpu::BufferAddress,
            usage:              self.usage,
            mapped_at_creation: false,
          }),
          reallocate_length,
        ));
      }
      None => {}
    }
    let buffer = &self.buffer_and_len.as_ref().unwrap().0;
    if !self.contents.is_empty() {
      let slice: &[u8] = unsafe { slice_as_u8_slice(&self.contents) };
      queue.write_buffer(buffer, 0, slice);
      slice.len()
    } else {
      0
    }
  }

  pub fn get_slice<'a>(&'a self) -> Option<BufferSlice<'a>> {
    match &self.buffer_and_len {
      Some((buffer, _)) => Some(buffer.slice(..)),
      None => None,
    }
  }

  #[inline]
  pub fn clear(&mut self) {
    self.contents.clear();
  }

  #[inline]
  pub fn push(&mut self, value: T) {
    self.contents.push(value);
  }

  #[inline]
  pub fn append(&mut self, values: &mut Vec<T>) {
    self.contents.append(values);
  }

  #[inline]
  pub fn len(&self) -> usize {
    self.contents.len()
  }
}

#[derive(Clone)]
pub struct RenderBufferOptions {
  pub format:                     wgpu::TextureFormat,
  pub sampler:                    Arc<wgpu::Sampler>,
  pub textured_bind_group_layout: Arc<wgpu::BindGroupLayout>,
  pub msaa_count:                 u32,
  pub mipmap_count:               u32,
}

pub struct RenderBufferContents {
  pub size:              (u32, u32),
  pub msaa_intermediate: Option<wgpu::Texture>,
  pub color:             wgpu::Texture,
  pub depth:             wgpu::Texture,
  pub bind_group:        wgpu::BindGroup,
}

pub struct RenderBuffer {
  pub label:    String,
  pub options:  RenderBufferOptions,
  pub contents: Option<RenderBufferContents>,
}

impl RenderBuffer {
  pub fn new(label: String, options: RenderBufferOptions) -> Self {
    Self {
      label,
      options,
      contents: None,
    }
  }

  pub fn get_buffers(
    &mut self,
    device: &wgpu::Device,
    new_size: (u32, u32),
  ) -> &RenderBufferContents {
    let reallocate = match self.contents.as_ref() {
      Some(RenderBufferContents { size, .. }) if *size == new_size => false,
      _ => true,
    };
    if reallocate {
      if let Some(contents) = self.contents.take() {
        if let Some(msaa_intermediate) = contents.msaa_intermediate {
          msaa_intermediate.destroy();
        }
        contents.color.destroy();
        contents.depth.destroy();
      }
      let make_tex = |format, label, sample_count, mip_level_count| {
        device.create_texture(&wgpu::TextureDescriptor {
          label: Some(label),
          size: wgpu::Extent3d {
            width:                 new_size.0,
            height:                new_size.1,
            depth_or_array_layers: 1,
          },
          mip_level_count,
          sample_count,
          dimension: wgpu::TextureDimension::D2,
          format,
          usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING,
          view_formats: &[], // FIXME: put in formats
        })
      };
      let label_color = format!("{}-color", self.label);
      let label_depth = format!("{}-depth", self.label);
      let msaa_intermediate = match self.options.msaa_count > 1 {
        true => Some(make_tex(
          self.options.format,
          &label_color,
          self.options.msaa_count,
          self.options.mipmap_count,
        )),
        false => None,
      };
      let color = make_tex(
        self.options.format,
        &label_color,
        self.options.msaa_count,
        self.options.mipmap_count,
      );
      let depth =
        make_tex(wgpu::TextureFormat::Depth32Float, &label_depth, self.options.msaa_count, 1);
      let texture_view = color.create_view(&wgpu::TextureViewDescriptor::default());
      let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout:  &self.options.textured_bind_group_layout,
        entries: &[
          wgpu::BindGroupEntry {
            binding:  0,
            resource: wgpu::BindingResource::TextureView(&texture_view),
          },
          wgpu::BindGroupEntry {
            binding:  1,
            resource: wgpu::BindingResource::Sampler(&self.options.sampler),
          },
        ],
        label:   Some(&format!("{}-bind-group", self.label)),
      });
      self.contents = Some(RenderBufferContents {
        size: new_size,
        msaa_intermediate,
        color,
        depth,
        bind_group,
      });
    }
    &self.contents.as_ref().unwrap()
  }
}

#[derive(Clone)]
pub struct CameraSettings {
  pub screen_size: (u32, u32),
  pub position:    Point3<f32>,
  pub heading:     f32,
  pub pitch:       f32,
}

impl CameraSettings {
  pub fn new() -> Self {
    Self {
      screen_size: (800, 600),
      position:    Point3::origin(),
      heading:     0.0,
      pitch:       0.0,
    }
  }

  pub fn aspect_ratio(&self) -> f32 {
    self.screen_size.0 as f32 / self.screen_size.1 as f32
  }

  pub fn eye_vector(&self) -> Vector3<f32> {
    let x = self.heading.cos() * self.pitch.cos();
    let y = self.pitch.sin();
    let z = self.heading.sin() * self.pitch.cos();
    Vector3::new(x, y, z)
  }

  pub fn right_vector(&self) -> Vector3<f32> {
    Vector3::new(self.heading.sin(), 0.0, -self.heading.cos())
  }
}

pub struct UniformsBuffer {
  pub uniforms:          Uniforms,
  pub uniforms_buffer:   wgpu::Buffer,
  pub bind_group_layout: wgpu::BindGroupLayout,
  pub bind_group:        wgpu::BindGroup,
}

impl UniformsBuffer {
  pub fn new(label: &str, device: &wgpu::Device) -> Self {
    let uniforms = Uniforms::new();
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label:    Some(label),
      contents: unsafe { ref_as_u8_slice(&uniforms) },
      usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      entries: &[wgpu::BindGroupLayoutEntry {
        binding:    0,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty:         wgpu::BindingType::Buffer {
          ty:                 wgpu::BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size:   None,
        },
        count:      None,
      }],
      label:   Some(label),
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout:  &bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding:  0,
        resource: uniforms_buffer.as_entire_binding(),
      }],
      label:   Some(label),
    });
    Self {
      uniforms,
      uniforms_buffer,
      bind_group_layout,
      bind_group,
    }
  }

  pub fn write_buffer(&self, queue: &wgpu::Queue) {
    queue.write_buffer(&self.uniforms_buffer, 0, unsafe { ref_as_u8_slice(&self.uniforms) });
  }
}

pub struct GeometryBuffer<V = Vertex> {
  pub vertex_buffer: ResizingBuffer<V>,
  pub index_buffer:  ResizingBuffer<u32>,
}

impl<V> GeometryBuffer<V> {
  pub fn new(name: &str) -> Self {
    Self {
      vertex_buffer: ResizingBuffer::new(
        name.to_string(),
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
      ),
      index_buffer:  ResizingBuffer::new(
        name.to_string(),
        wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
      ),
    }
  }

  pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> usize {
    self.vertex_buffer.update(device, queue) + self.index_buffer.update(device, queue)
  }

  pub fn clear(&mut self) {
    self.vertex_buffer.clear();
    self.index_buffer.clear();
  }
}

impl GeometryBuffer<Vertex> {
  pub fn add_line(&mut self, a: Point3<f32>, b: Point3<f32>, color: [f32; 3]) {
    let index = self.vertex_buffer.len() as u32;
    self.vertex_buffer.push(Vertex {
      position: [a.x, a.y, a.z],
      color,
      uv: [0.0, 0.0],
    });
    self.vertex_buffer.push(Vertex {
      position: [b.x, b.y, b.z],
      color,
      uv: [0.0, 0.0],
    });
    self.index_buffer.push(index);
    self.index_buffer.push(index + 1);
  }
}

pub struct GpuDataTextureBuffer {
  pub data: Vec<u32>,
  pub ptr: usize,
}

impl GpuDataTextureBuffer {
  pub fn new() -> Self {
    Self {
      data: vec![0; DATA_TEXTURE_SIZE * DATA_TEXTURE_SIZE],
      ptr:  0,
    }
  }

  pub fn push(&mut self, value: u32) {
    self.data[self.ptr] = value;
    self.ptr += 1;
  }

  pub fn reset(&mut self) {
    self.ptr = 0;
  }

  /// Returns (rows of image data, slice of bytes).
  pub fn get_write_info(&self) -> (u32, &[u8]) {
    // Round up to a full row.
    let rows = (self.ptr + DATA_TEXTURE_SIZE - 1) / DATA_TEXTURE_SIZE;
    let bytes = 4 * rows * DATA_TEXTURE_SIZE;
    (rows as u32, unsafe {
      std::slice::from_raw_parts(self.data.as_ptr() as *const u8, bytes as usize)
    })
  }
}

static SHADER_SOURCE: &str = include_str!("shaders.wgsl");

pub struct RenderData {
  pub main_uniforms:      UniformsBuffer,
  pub triangles_geo:      GeometryBuffer,
  pub lines_geo:          GeometryBuffer,
  pub rays_geo:           GeometryBuffer,
  //pub paths_geo:          GeometryBuffer<RayPathVertex>,
  pub voxel_template:     GeometryBuffer,
  pub voxel_instances:    ResizingBuffer<VoxelInstance>,
  pub shadow_voxel_instances: ResizingBuffer<VoxelInstance>,
  pub voxel_data:         GpuDataTextureBuffer,
  pub voxel_data_texture: wgpu::Texture,
  pub voxel_data_texture_view:    wgpu::TextureView,
  pub data_texture_bind_group:    wgpu::BindGroup,
  pub triangles_pipeline: wgpu::RenderPipeline,
  pub lines_pipeline:     wgpu::RenderPipeline,
  //pub ray_paths_pipeline: wgpu::RenderPipeline,
  pub voxels_pipeline:    wgpu::RenderPipeline,
  pub shadow_voxels_pipeline: wgpu::RenderPipeline,
  pub full_screen_pipeline: wgpu::RenderPipeline,
}

impl RenderData {
  pub fn new(cc: &eframe::CreationContext) -> Self {
    let wgpu_render_state = cc.wgpu_render_state.as_ref().unwrap();
    let device = &wgpu_render_state.device;
    let queue = &wgpu_render_state.queue;

    let mut voxel_template = GeometryBuffer::new("voxel-template");
    // We push all six faces separately.
    let mut push_face =
      |a: (i32, i32, i32), b: (i32, i32, i32), c: (i32, i32, i32), d: (i32, i32, i32)| {
        let base_index = voxel_template.vertex_buffer.len() as u32;
        voxel_template.vertex_buffer.push(Vertex {
          position: [a.0 as f32, a.1 as f32, a.2 as f32],
          color:    [1.0, 1.0, 1.0],
          uv:       [0.0, 0.0],
        });
        voxel_template.vertex_buffer.push(Vertex {
          position: [b.0 as f32, b.1 as f32, b.2 as f32],
          color:    [1.0, 1.0, 1.0],
          uv:       [1.0, 0.0],
        });
        voxel_template.vertex_buffer.push(Vertex {
          position: [c.0 as f32, c.1 as f32, c.2 as f32],
          color:    [1.0, 1.0, 1.0],
          uv:       [0.0, 1.0],
        });
        voxel_template.vertex_buffer.push(Vertex {
          position: [d.0 as f32, d.1 as f32, d.2 as f32],
          color:    [1.0, 1.0, 1.0],
          uv:       [1.0, 1.0],
        });
        for i in [0, 1, 2, 2, 1, 3] {
          voxel_template.index_buffer.push(base_index + i);
        }
      };
    push_face((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)); // front
    push_face((0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)); // back
    push_face((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)); // top
    push_face((0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)); // bottom
    push_face((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)); // left
    push_face((1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)); // right
    voxel_template.update(device, queue);

    let voxel_data_texture = device.create_texture(&wgpu::TextureDescriptor {
      size:            wgpu::Extent3d {
        width:                 DATA_TEXTURE_SIZE as u32,
        height:                DATA_TEXTURE_SIZE as u32,
        depth_or_array_layers: 1,
      },
      mip_level_count: 1,
      sample_count:    1,
      dimension:       wgpu::TextureDimension::D2,
      format:          wgpu::TextureFormat::R32Uint,
      usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
      label:           Some("voxel_data_texture"),
      view_formats:    &[], // FIXME
    });
    let voxel_data_texture_view = voxel_data_texture.create_view(&wgpu::TextureViewDescriptor {
      format: Some(wgpu::TextureFormat::R32Uint),
      ..Default::default()
    });
    let data_texture_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
          binding:    0,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty:         wgpu::BindingType::Texture {
            multisampled:   false,
            view_dimension: wgpu::TextureViewDimension::D2,
            sample_type:    wgpu::TextureSampleType::Uint,
          },
          count:      None,
        }],
        label:   Some("data_texture_bind_group_layout"),
      });
    let data_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout:  &data_texture_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding:  0,
        resource: wgpu::BindingResource::TextureView(&voxel_data_texture_view),
      }],
      label:   Some("data_texture_bind_group"),
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label:  None,
      source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SOURCE)),
    });

    let main_uniforms = UniformsBuffer::new("main-uniforms", device);

    let default_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label:                None,
      bind_group_layouts:   &[&main_uniforms.bind_group_layout],
      push_constant_ranges: &[],
    });
    let data_texture_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label:                None,
      bind_group_layouts:   &[&main_uniforms.bind_group_layout, &data_texture_bind_group_layout],
      push_constant_ranges: &[],
    });

    let no_blending: wgpu::ColorTargetState = wgpu_render_state.target_format.into();
    let mut alpha_blending = no_blending.clone();
    alpha_blending.blend = Some(wgpu::BlendState::ALPHA_BLENDING);

    macro_rules! make_pipeline {
      (
        layout = $layout:expr;
        vertex_buffers = $vertex_buffers:expr;
        vertex = $vertex:literal;
        fragment = $fragment:literal;
        topology = $topology:expr;
        blend_mode = $blend_mode:expr;
        $( depth_compare = $depth_compare:expr; )?
        $( depth_write = $depth_write:expr; )?
      ) => {{
        #[allow(unused_mut, unused_assignments)]
        let mut depth_compare_enabled = true;
        #[allow(unused_mut, unused_assignments)]
        let mut depth_write_enabled = true;
        $( depth_compare_enabled = $depth_compare; )?
        $( depth_write_enabled = $depth_write; )?
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
          label:         None,
          layout:        Some(&$layout),
          vertex:        wgpu::VertexState {
            module:      &shader,
            entry_point: $vertex,
            buffers:     &$vertex_buffers,
          },
          fragment:      Some(wgpu::FragmentState {
            module:      &shader,
            entry_point: $fragment,
            targets:     &[Some($blend_mode.clone())],
          }),
          primitive:     wgpu::PrimitiveState {
            topology:           $topology,
            strip_index_format: None,
            front_face:         wgpu::FrontFace::Ccw,
            cull_mode:          None,
            polygon_mode:       wgpu::PolygonMode::Fill,
            unclipped_depth:    false,
            conservative:       false,
          },
          depth_stencil: Some(wgpu::DepthStencilState {
            format:              wgpu::TextureFormat::Depth32Float,
            depth_write_enabled,
            depth_compare:       match depth_compare_enabled {
              true => wgpu::CompareFunction::Less,
              false => wgpu::CompareFunction::Always,
            },
            stencil:             wgpu::StencilState::default(),
            bias:                wgpu::DepthBiasState::default(),
          }),
          multisample:   wgpu::MultisampleState {
            count:                     1, //$msaa_samples,
            mask:                      !0,
            alpha_to_coverage_enabled: false,
          },
          multiview:     None,
        })
      }};
    }

    let triangles_pipeline = make_pipeline!(
      layout = default_pipeline_layout;
      vertex_buffers = [Vertex::desc()];
      vertex = "vertex_shader_main";
      fragment = "solid_fragment_shader_main";
      topology = wgpu::PrimitiveTopology::TriangleList;
      blend_mode = no_blending;
    );
    let lines_pipeline = make_pipeline!(
      layout = default_pipeline_layout;
      vertex_buffers = [Vertex::desc()];
      vertex = "vertex_shader_main";
      fragment = "solid_fragment_shader_main";
      topology = wgpu::PrimitiveTopology::LineList;
      blend_mode = no_blending;
    );
    // let ray_paths_pipeline = make_pipeline!(
    //   layout = default_pipeline_layout;
    //   vertex_buffers = [RayPathVertex::desc()];
    //   vertex = "ray_paths_vertex_shader_main";
    //   fragment = "solid_fragment_shader_main";
    //   topology = wgpu::PrimitiveTopology::LineList;
    // );
    let voxels_pipeline = make_pipeline!(
      layout = default_pipeline_layout;
      vertex_buffers = [Vertex::desc(), VoxelInstance::desc()];
      vertex = "voxel_vertex_shader_main";
      fragment = "edge_highlight_fragment_shader_main";
      topology = wgpu::PrimitiveTopology::TriangleList;
      blend_mode = no_blending;
    );
    let shadow_voxels_pipeline = make_pipeline!(
      layout = default_pipeline_layout;
      vertex_buffers = [Vertex::desc(), VoxelInstance::desc()];
      vertex = "voxel_vertex_shader_main";
      fragment = "shadow_edge_highlight_fragment_shader_main";
      topology = wgpu::PrimitiveTopology::TriangleList;
      blend_mode = alpha_blending;
      depth_compare = true;
      depth_write = false;
    );
    let full_screen_pipeline = make_pipeline!(
      layout = data_texture_pipeline_layout;
      vertex_buffers = [];
      vertex = "full_screen_vertex_shader_main";
      fragment = "full_screen_fragment_shader_main";
      topology = wgpu::PrimitiveTopology::TriangleStrip;
      blend_mode = no_blending;
    );

    let mut rays_geo = GeometryBuffer::new("rays-geo");
    rays_geo.update(device, queue);

    Self {
      main_uniforms,
      triangles_geo: GeometryBuffer::new("triangles-geo"),
      lines_geo: GeometryBuffer::new("lines-geo"),
      rays_geo,
      //paths_geo: GeometryBuffer::new("paths-geo"),
      voxel_template,
      voxel_instances: ResizingBuffer::new(
        "voxel-instances".to_string(),
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
      ),
      shadow_voxel_instances: ResizingBuffer::new(
        "shadow-voxel-instances".to_string(),
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
      ),
      voxel_data: GpuDataTextureBuffer::new(),
      voxel_data_texture,
      voxel_data_texture_view,
      data_texture_bind_group,
      triangles_pipeline,
      lines_pipeline,
      //ray_paths_pipeline,
      voxels_pipeline,
      shadow_voxels_pipeline,
      full_screen_pipeline,
    }
  }

  pub fn flush_data_texture(&self, queue: &wgpu::Queue) {
    let (row_count, bytes) = self.voxel_data.get_write_info();
    if row_count == 0 {
      return;
    }
    queue.write_texture(
      wgpu::ImageCopyTexture {
        texture:   &self.voxel_data_texture,
        mip_level: 0,
        origin:    wgpu::Origin3d::ZERO,
        aspect:    wgpu::TextureAspect::All,
      },
      bytes,
      wgpu::ImageDataLayout {
        offset:         0,
        bytes_per_row:  Some(4 * DATA_TEXTURE_SIZE as u32),
        rows_per_image: Some(row_count),
      },
      wgpu::Extent3d {
        width:                 DATA_TEXTURE_SIZE as u32,
        height:                row_count,
        depth_or_array_layers: 1,
      },
    );
  }
}

// ==================== web handle ====================

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  pub fn log(s: &str);
}

#[cfg(target_arch = "wasm32")]
use eframe::WebRunner;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
#[wasm_bindgen]
pub struct WebHandle {
  runner: WebRunner,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WebHandle {
  #[wasm_bindgen(constructor)]
  pub fn new() -> Self {
    Self {
      runner: WebRunner::new(),
    }
  }

  #[wasm_bindgen]
  pub async fn start(&self, canvas_id: &str) -> Result<(), wasm_bindgen::JsValue> {
    let mut web_options = eframe::WebOptions::default();
    web_options.depth_buffer = 32;
    web_options.wgpu_options.supported_backends = wgpu::Backends::GL;
    web_sys::window()
      .unwrap()
      .document()
      .unwrap()
      .get_element_by_id("loadingMessage")
      .unwrap()
      .set_attribute("style", "display: none")
      .unwrap();
    log("Launching app from WASM");
    self
      .runner
      .start(canvas_id, web_options, Box::new(|cc| Box::new(ReactorSimulatorApp::new(cc))))
      .await
  }

  #[wasm_bindgen]
  pub fn destroy(&self) {
    self.runner.destroy();
  }

  // /// Example on how to call into your app from JavaScript.
  // #[wasm_bindgen]
  // pub fn example(&self) {
  //     if let Some(app) = self.runner.app_mut::<EframeApp>() {
  //         app.example();
  //     }
  // }

  /// The JavaScript can check whether or not your app has crashed:
  #[wasm_bindgen]
  pub fn has_panicked(&self) -> bool {
    self.runner.has_panicked()
  }

  #[wasm_bindgen]
  pub fn panic_message(&self) -> Option<String> {
    self.runner.panic_summary().map(|s| s.message())
  }

  #[wasm_bindgen]
  pub fn panic_callstack(&self) -> Option<String> {
    self.runner.panic_summary().map(|s| s.callstack())
  }
}

// ==================== eframe app ====================

pub struct ReactorSimulatorApp {
  locked_state: Arc<Mutex<GameState>>,
}

impl ReactorSimulatorApp {
  pub fn new(cc: &eframe::CreationContext) -> Self {
    let limits = cc.wgpu_render_state.as_ref().unwrap().adapter.limits();
    //log(&format!("Limits: {:#?}", limits));
    println!("Limits: {:#?}", limits);
    let locked_state = Arc::new(Mutex::new(GameState::new()));
    let wgpu_render_state = cc.wgpu_render_state.as_ref().unwrap();
    let mut w = wgpu_render_state.renderer.write();
    w.paint_callback_resources.insert(RenderData::new(cc));
    Self { locked_state }
  }
}

impl eframe::App for ReactorSimulatorApp {
  fn update(&mut self, egui_ctx: &egui::Context, frame: &mut eframe::Frame) {
    egui_ctx.set_visuals(egui::Visuals::dark());
    let dt = egui_ctx.input(|inp| inp.stable_dt.clamp(0.0, 0.15));

    // Draw all of the mundate GUI elements, and update
    {
      let mut guard = self.locked_state.lock().unwrap();
      guard.update(dt);
      guard.draw_main_gui(egui_ctx, frame);
    }

    // Perform custom painting into an allocated rect.
    let locked_state_prepare = Arc::clone(&self.locked_state);
    let locked_state_paint = Arc::clone(&self.locked_state);
    egui::CentralPanel::default()
      //.frame(egui::Frame::none().fill(egui::Color32::from_rgb(80, 80, 250)))
      .frame(egui::Frame::none().fill(egui::Color32::from_rgb(10, 10, 10)))
      .show(egui_ctx, |ui| {
        let (id, allocated_rect) = ui.allocate_space(ui.available_size());
        let response = ui.interact(allocated_rect, id, egui::Sense::click_and_drag());
        {
          let mut guard = locked_state_paint.lock().unwrap();
          guard.response(egui_ctx, allocated_rect, response);
        }

        let cb = egui_wgpu::CallbackFn::new()
          .prepare(move |device, queue, _encoder, paint_callback_resources| {
            let render_data: &mut RenderData = paint_callback_resources.get_mut().unwrap();
            let mut guard = locked_state_prepare.lock().unwrap();
            // To avoid some divide by zeros and wgpu errors we clamp the size here.
            let mut rect_with_min_size = allocated_rect;
            if rect_with_min_size.width() < 2.0 {
              rect_with_min_size.set_width(2.0);
            }
            if rect_with_min_size.height() < 2.0 {
              rect_with_min_size.set_height(2.0);
            }
            guard.camera.screen_size =
              (rect_with_min_size.width() as u32, rect_with_min_size.height() as u32);
            guard.prepare(render_data, device, queue);
            Vec::new()
          })
          .paint(move |_info, render_pass, paint_callback_resources| {
            let guard = locked_state_paint.lock().unwrap();
            let render_data: &RenderData = paint_callback_resources.get().unwrap();
            guard.paint(render_data, render_pass);
          });
        ui.painter().add(egui::PaintCallback {
          rect:     allocated_rect,
          callback: Arc::new(cb),
        });
      });

    // Rerender every frame (suppress the default eframe laziness).
    egui_ctx.request_repaint();
  }
}
