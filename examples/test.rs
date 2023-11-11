use enum_map::Enum;
use peter_engine::{
  eframe, egui,
  graphics::{RenderData, ResizingBuffer, ResourceKey, Vertex},
  launch, wgpu, PeterEngineApp,
};

struct TestApp {}

#[derive(Enum)]
enum Textures {
  A,
}

impl ResourceKey for Textures {
  type Output = wgpu::TextureView;

  fn load<App: PeterEngineApp>(self, rd: &mut RenderData<App>) -> Self::Output {
    match self {
      Self::A =>
        rd.load_texture(&[
          0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
          0x52, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x03, 0x00, 0x00, 0x00, 0x66,
          0xBC, 0x3A, 0x25, 0x00, 0x00, 0x00, 0x03, 0x50, 0x4C, 0x54, 0x45, 0xB5, 0xD0, 0xD0, 0x63,
          0x04, 0x16, 0xEA, 0x00, 0x00, 0x00, 0x1F, 0x49, 0x44, 0x41, 0x54, 0x68, 0x81, 0xED, 0xC1,
          0x01, 0x0D, 0x00, 0x00, 0x00, 0xC2, 0xA0, 0xF7, 0x4F, 0x6D, 0x0E, 0x37, 0xA0, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xBE, 0x0D, 0x21, 0x00, 0x00, 0x01, 0x9A, 0x60, 0xE1,
          0xD5, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ])
        .1,
    }
  }
}

#[derive(Enum)]
enum VertexBuffers {
  A,
}

impl ResourceKey for VertexBuffers {
  type Output = ResizingBuffer<Vertex>;

  fn load<App: PeterEngineApp>(self, rd: &mut RenderData<App>) -> Self::Output {
    match self {
      Self::A => ResizingBuffer::new("VertexBuffers::A", wgpu::BufferUsages::VERTEX),
    }
  }
}

impl PeterEngineApp for TestApp {
  const WINDOW_TITLE: &'static str = "Test App";
  const SHADER_SOURCE: &'static str = r#"

  "#;

  fn init(&mut self, _cc: &eframe::CreationContext, rd: &mut RenderData<Self>) {
    rd.load_resources::<Textures>();
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
    rd: &mut RenderData<Self>,
    encoder: &mut wgpu::CommandEncoder,
  ) -> Vec<wgpu::CommandBuffer> {
    let r = rd.get_resource(Textures::A);
    Vec::new()
  }

  fn paint(
    &mut self,
    rd: &RenderData<Self>,
    info: eframe::epaint::PaintCallbackInfo,
    render_pass: &mut wgpu::RenderPass,
  ) {
  }
}

fn main() {
  launch(TestApp {}).unwrap();
}
