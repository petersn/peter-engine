use std::sync::Arc;

use anymap::AnyMap;
use enum_map::{EnumArray, EnumMap};
use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3};
use wgpu::util::DeviceExt;
use wgpu::BufferSlice;

use crate::mipmapping::MipMapGen;
use crate::PeterEngineApp;

const DATA_TEXTURE_SIZE: usize = 1024;

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
  pub color:    [f32; 4],
  pub uv:       [f32; 2],
}

impl Vertex {
  const ATTRIBS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4, 2 => Float32x2];

  pub fn desc() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
      array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
      step_mode:    wgpu::VertexStepMode::Vertex,
      attributes:   &Self::ATTRIBS,
    }
  }
}

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
  pub fn new(label: &str, usage: wgpu::BufferUsages) -> Self {
    Self {
      label: label.to_string(),
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
        name,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
      ),
      index_buffer:  ResizingBuffer::new(
        name,
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
  pub fn add_line(&mut self, a: Point3<f32>, b: Point3<f32>, color: [f32; 4]) {
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
  pub ptr:  usize,
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

pub static SHADER_PRELUDE: &str = include_str!("shaders.wgsl");

// struct PipelineDesc {
//   //layout:
// }

// pub trait PipelinesEnum: EnumArray<wgpu::RenderPipeline> + Send + Sync {
//   fn describe(self) -> PipelineDesc;
// }

// pub trait TexturesEnum: EnumArray<wgpu::Texture> + Send + Sync {
//   fn data(self) -> &'static [u8];
// }

pub trait ResourceKey: Sized + Send + Sync + 'static + EnumArray<Self::Output> {
  type Output: Sized + Send + Sync + 'static;

  fn load<App: PeterEngineApp>(self, rd: &mut RenderData<App>) -> Self::Output;
}

pub struct RenderData<App: PeterEngineApp + ?Sized> {
  pub main_uniforms:      UniformsBuffer,
  pub device:             Arc<wgpu::Device>,
  pub queue:              Arc<wgpu::Queue>,
  pub shader:             wgpu::ShaderModule,
  // pub main_data:               GpuDataTextureBuffer,
  // pub main_data_texture:       wgpu::Texture,
  // pub main_data_texture_view:  wgpu::TextureView,
  // pub data_texture_bind_group: wgpu::BindGroup,
  pub mipmap_gen:         MipMapGen,
  pub pixel_perfect_size: (u32, u32),
  pub resources:          AnyMap,
  pub _phantom:           std::marker::PhantomData<App>,
}

// unsafe impl<App: PeterEngineApp> Send for RenderData<App> {}
// unsafe impl<App: PeterEngineApp> Sync for RenderData<App> {}

impl<App: PeterEngineApp> RenderData<App> {
  pub fn load_resources<K: ResourceKey>(&mut self) {
    if self.resources.contains::<EnumMap<K, K::Output>>() {
      panic!("Resource type already loaded");
    }
    let mapping = EnumMap::from_fn(|key: K| key.load(self));
    self.resources.insert::<EnumMap<K, K::Output>>(mapping);
  }

  pub fn get_resource<K: ResourceKey>(&self, key: K) -> &K::Output {
    match self.resources.get::<EnumMap<K, K::Output>>() {
      Some(mapping) => &mapping[key],
      None => panic!("Resource not loaded: {}", std::any::type_name::<K>()),
    }
  }

  pub fn get_resource_mut<K: ResourceKey>(&mut self, key: K) -> &mut K::Output {
    match self.resources.get_mut::<EnumMap<K, K::Output>>() {
      Some(mapping) => &mut mapping[key],
      None => panic!("Resource not loaded: {}", std::any::type_name::<K>()),
    }
  }

  pub fn new(cc: &eframe::CreationContext) -> Self {
    let wgpu_render_state = cc.wgpu_render_state.as_ref().unwrap();
    let device = Arc::clone(&wgpu_render_state.device);
    let queue = Arc::clone(&wgpu_render_state.queue);

    // let main_data_texture = device.create_texture(&wgpu::TextureDescriptor {
    //   size:            wgpu::Extent3d {
    //     width:                 DATA_TEXTURE_SIZE as u32,
    //     height:                DATA_TEXTURE_SIZE as u32,
    //     depth_or_array_layers: 1,
    //   },
    //   mip_level_count: 1,
    //   sample_count:    1,
    //   dimension:       wgpu::TextureDimension::D2,
    //   format:          wgpu::TextureFormat::R32Uint,
    //   usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    //   label:           Some("main_data_texture"),
    //   view_formats:    &[], // FIXME
    // });
    // let main_data_texture_view = main_data_texture.create_view(&wgpu::TextureViewDescriptor {
    //   format: Some(wgpu::TextureFormat::R32Uint),
    //   ..Default::default()
    // });
    // let data_texture_bind_group_layout =
    //   device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    //     entries: &[wgpu::BindGroupLayoutEntry {
    //       binding:    0,
    //       visibility: wgpu::ShaderStages::FRAGMENT,
    //       ty:         wgpu::BindingType::Texture {
    //         multisampled:   false,
    //         view_dimension: wgpu::TextureViewDimension::D2,
    //         sample_type:    wgpu::TextureSampleType::Uint,
    //       },
    //       count:      None,
    //     }],
    //     label:   Some("data_texture_bind_group_layout"),
    //   });
    // let data_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    //   layout:  &data_texture_bind_group_layout,
    //   entries: &[wgpu::BindGroupEntry {
    //     binding:  0,
    //     resource: wgpu::BindingResource::TextureView(&main_data_texture_view),
    //   }],
    //   label:   Some("data_texture_bind_group"),
    // });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label:  None,
      source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(App::SHADER_SOURCE)),
    });

    let main_uniforms = UniformsBuffer::new("main_uniforms", &device);

    let default_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label:                None,
      bind_group_layouts:   &[&main_uniforms.bind_group_layout],
      push_constant_ranges: &[],
    });
    // let data_texture_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    //   label:                None,
    //   bind_group_layouts:   &[&main_uniforms.bind_group_layout, &data_texture_bind_group_layout],
    //   push_constant_ranges: &[],
    // });

    let no_blending: wgpu::ColorTargetState = wgpu_render_state.target_format.into();
    let mut alpha_blending = no_blending.clone();
    alpha_blending.blend = Some(wgpu::BlendState::ALPHA_BLENDING);

    macro_rules! make_pipeline {
      (
        layout = $layout:expr;
        vertex_buffers = $vertex_buffers:expr;
        vertex = $vertex:expr;
        fragment = $fragment:expr;
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

    let mipmap_gen = MipMapGen::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // let pipelines = EnumMap::from_fn(|pipeline_desc| {
    //   todo!()
    // });
    // let textures = EnumMap::from_fn(|texture_desc| {
    //   todo!()
    // });
    // let buffers = EnumMap::from_fn(|buffer_desc| {
    //   todo!()
    // });

    Self {
      main_uniforms,
      device,
      queue,
      shader,
      // main_data: GpuDataTextureBuffer::new(),
      // main_data_texture,
      // main_data_texture_view,
      // data_texture_bind_group,
      mipmap_gen,
      pixel_perfect_size: (1, 1),
      resources: AnyMap::new(),
      // pipelines,
      // textures,
      _phantom: std::marker::PhantomData,
    }
  }

  pub fn load_texture(&self, bytes: &[u8]) -> (wgpu::Texture, wgpu::TextureView) {
    let diffuse_image = image::load_from_memory(bytes).unwrap();
    let diffuse_rgba = diffuse_image.to_rgba8();
    let dimensions = diffuse_rgba.dimensions();
    let texture_size = wgpu::Extent3d {
      width:                 dimensions.0,
      height:                dimensions.1,
      depth_or_array_layers: 1,
    };
    let mip_level_count = 4;
    let diffuse_texture = self.device.create_texture(&wgpu::TextureDescriptor {
      size: texture_size,
      mip_level_count,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: wgpu::TextureFormat::Rgba8UnormSrgb,
      usage: wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_DST
        | wgpu::TextureUsages::RENDER_ATTACHMENT,
      label: Some("font_texture"),
      view_formats: &[],
    });
    self.queue.write_texture(
      wgpu::ImageCopyTexture {
        texture:   &diffuse_texture,
        mip_level: 0,
        origin:    wgpu::Origin3d::ZERO,
        aspect:    wgpu::TextureAspect::All,
      },
      &diffuse_rgba,
      wgpu::ImageDataLayout {
        offset:         0,
        bytes_per_row:  Some(4 * dimensions.0),
        rows_per_image: Some(dimensions.1),
      },
      texture_size,
    );
    let mut mipmap_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
      label: Some("mipmap_gen"),
    });
    self.mipmap_gen.generate_mipmaps(
      &mut mipmap_encoder,
      &self.device,
      &diffuse_texture,
      mip_level_count,
    );
    self.queue.submit(Some(mipmap_encoder.finish()));
    let view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
    (diffuse_texture, view)
  }

  // pub fn flush_data_texture(&self, queue: &wgpu::Queue) {
  //   let (row_count, bytes) = self.main_data.get_write_info();
  //   if row_count == 0 {
  //     return;
  //   }
  //   queue.write_texture(
  //     wgpu::ImageCopyTexture {
  //       texture:   &self.main_data_texture,
  //       mip_level: 0,
  //       origin:    wgpu::Origin3d::ZERO,
  //       aspect:    wgpu::TextureAspect::All,
  //     },
  //     bytes,
  //     wgpu::ImageDataLayout {
  //       offset:         0,
  //       bytes_per_row:  Some(4 * DATA_TEXTURE_SIZE as u32),
  //       rows_per_image: Some(row_count),
  //     },
  //     wgpu::Extent3d {
  //       width:                 DATA_TEXTURE_SIZE as u32,
  //       height:                row_count,
  //       depth_or_array_layers: 1,
  //     },
  //   );
  // }
}
