use enum_map::Enum;
use peter_engine::{eframe, egui, graphics::RenderData, wgpu, PeterEngineApp, launch};
use wgpu::util::make_spirv_raw;

struct TestApp {}

// #[derive(Enum)]
// enum Textures {
//   A,
// }

// impl ResourceKey for Textures {
//   type Output = wgpu::TextureView;

//   fn load<App: PeterEngineApp>(self, rd: &mut RenderData<App>) -> Self::Output {
//     match self {
//       Self::A => rd.load_texture(&[
//         0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
//         0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x03, 0x00, 0x00, 0x00, 0x66, 0xBC, 0x3A,
//         0x25, 0x00, 0x00, 0x00, 0x03, 0x50, 0x4C, 0x54, 0x45, 0xB5, 0xD0, 0xD0, 0x63, 0x04, 0x16, 0xEA,
//         0x00, 0x00, 0x00, 0x1F, 0x49, 0x44, 0x41, 0x54, 0x68, 0x81, 0xED, 0xC1, 0x01, 0x0D, 0x00, 0x00,
//         0x00, 0xC2, 0xA0, 0xF7, 0x4F, 0x6D, 0x0E, 0x37, 0xA0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
//         0x00, 0xBE, 0x0D, 0x21, 0x00, 0x00, 0x01, 0x9A, 0x60, 0xE1, 0xD5, 0x00, 0x00, 0x00, 0x00, 0x49,
//         0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
//       ]).1,
//     }
//   }
// }

struct AppResources {
  main_pipeline: wgpu::RenderPipeline,
}

impl PeterEngineApp for TestApp {
  const WINDOW_TITLE: &'static str = "Test App";
  const SHADER_SOURCE: &'static str = r#"

  "#;

  type RenderResources = AppResources;

  fn init(&mut self, cc: &eframe::CreationContext, rd: &mut RenderData) -> Self::RenderResources {
    let no_blending: wgpu::ColorTargetState = cc.wgpu_render_state.as_ref().unwrap().target_format.into();
    let mut alpha_blending = no_blending.clone();
    alpha_blending.blend = Some(wgpu::BlendState::ALPHA_BLENDING);

    let main_pipeline = peter_engine::make_pipeline!(
      render_data = rd;
      layout = rd.pipeline_layout;
      vertex_buffers = [];
      vertex = "foo";
      fragment = "bar";
      topology = wgpu::PrimitiveTopology::TriangleList;
      blend_mode = no_blending;
    );
    AppResources {
      main_pipeline,
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
    rd: &mut RenderData,
    _resources: &mut Self::RenderResources,
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
  ) -> Vec<wgpu::CommandBuffer> {
    Vec::new()
  }

  fn paint(
    &mut self,
    rd: &RenderData,
    _resources: &Self::RenderResources,
    info: eframe::epaint::PaintCallbackInfo,
    render_pass: &mut wgpu::RenderPass,
  ) {
  }
}

fn main() {
  launch(TestApp {}).unwrap();
}
