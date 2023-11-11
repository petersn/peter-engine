use peter_engine::{
  eframe, egui,
  graphics::{PipelineDesc, RenderData, Vertex, SHADER_PRELUDE, BindingDesc, GeometryBuffer},
  launch, wgpu, PeterEngineApp,
};

struct AppResources {
  main_pipeline: wgpu::RenderPipeline,
  trianges: GeometryBuffer,
}

struct TestApp {}

impl PeterEngineApp for TestApp {
  const WINDOW_TITLE: &'static str = "Test App";

  type RenderResources = AppResources;

  fn get_shader_source() -> String {
    format!(
      "{}\n{}",
      SHADER_PRELUDE,
      r#"
        @vertex
        fn vertex_shader(
          model: VertexInput,
        ) -> VertexOutput {
          var out: VertexOutput;
          out.color = model.color;
          out.uv = model.uv;
          out.clip_position = uniforms.transform_pvm * vec4<f32>(model.position, 1.0);
          return out;
        }
        @fragment
        fn solid_fragment_shader(in: VertexOutput) -> @location(0) vec4<f32> {
          return in.color;
        }
      "#,
    )
  }

  fn init(&mut self, _cc: &eframe::CreationContext, rd: &mut RenderData) -> Self::RenderResources {
    let main_pipeline = rd.create_pipeline(PipelineDesc {
      layout: vec![vec![BindingDesc::Uniforms]],
      vertex_buffers: vec![Vertex::desc()],
      vertex_shader: "vertex_shader",
      fragment_shader: "solid_fragment_shader",
      topology: wgpu::PrimitiveTopology::TriangleList,
      ..Default::default()
    });
    AppResources {
      main_pipeline,
      trianges: GeometryBuffer::new("triangles"),
    }
  }

  fn update(&mut self, egui_ctx: &egui::Context, frame: &mut eframe::Frame, dt: f32) {
    // Put an egui window up.
    egui::Window::new("Hello world").show(egui_ctx, |ui| {
      ui.label(format!("Hello world! ({:.1} FPS)", 1.0 / dt));
      if ui.button("Quit").clicked() {
        frame.close();
      }
    });
  }

  fn prepare(
    &mut self,
    _rd: &mut RenderData,
    resources: &mut Self::RenderResources,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    _encoder: &mut wgpu::CommandEncoder,
    _screen_size: (u32, u32),
  ) -> Vec<wgpu::CommandBuffer> {
    resources.trianges.clear();
    resources.trianges.vertex_buffer.extend_from_slice(&[
      Vertex {
        position: [0.0, 0.0, 0.0],
        color: [1.0, 0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
      },
      Vertex {
        position: [1.0, 0.0, 0.0],
        color: [0.0, 1.0, 0.0, 1.0],
        uv: [0.0, 0.0],
      },
      Vertex {
        position: [0.0, 1.0, 0.0],
        color: [0.0, 0.0, 1.0, 1.0],
        uv: [0.0, 0.0],
      },
    ]);
    resources.trianges.index_buffer.extend_from_slice(&[0, 1, 2]);
    resources.trianges.update(device, queue);
    Vec::new()
  }

  fn paint<'rp>(
    &mut self,
    rd: &'rp RenderData,
    resources: &'rp Self::RenderResources,
    _info: eframe::epaint::PaintCallbackInfo,
    render_pass: &mut wgpu::RenderPass<'rp>,
  ) {
    render_pass.set_pipeline(&resources.main_pipeline);
    render_pass.set_bind_group(0, &rd.main_uniforms.bind_group, &[]);
    render_pass.set_vertex_buffer(0, resources.trianges.vertex_buffer.get_slice().unwrap());
    render_pass.set_index_buffer(resources.trianges.index_buffer.get_slice().unwrap(), wgpu::IndexFormat::Uint32);
    render_pass.draw_indexed(0..resources.trianges.index_buffer.len() as u32, 0, 0..1);
  }
}

fn main() {
  launch(TestApp {}).unwrap();
}
