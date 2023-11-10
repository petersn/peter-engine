// Re-export our main dependencies.
pub use eframe;
pub use eframe::{wgpu, egui};
pub use nalgebra;

use std::sync::{Arc, Mutex};

use graphics::RenderData;
use eframe::egui_wgpu::{CallbackTrait, CallbackResources, Callback};

pub mod graphics;

pub trait PeterEngineApp: Send + 'static {
  fn draw_main_gui(&mut self, egui_ctx: &egui::Context, frame: &mut eframe::Frame, dt: f32);
  fn central_panel_input(&mut self, egui_ctx: &egui::Context, response: egui::Response, allocated_rect: &egui::Rect);
  fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, render_data: &mut RenderData) -> Vec<wgpu::CommandBuffer>;
  fn paint(&mut self, info: eframe::epaint::PaintCallbackInfo, render_pass: &mut wgpu::RenderPass, render_data: &RenderData);
}

struct EframeApp<GameState> {
  locked_state: Arc<Mutex<GameState>>,
}

impl<GameState> EframeApp<GameState> {
  fn new(
    game_state: GameState,
    cc: &eframe::CreationContext,
  ) -> Self {
    // let limits = cc.wgpu_render_state.as_ref().unwrap().adapter.limits();
    let locked_state = Arc::new(Mutex::new(game_state));
    let wgpu_render_state = cc.wgpu_render_state.as_ref().unwrap();
    let mut w = wgpu_render_state.renderer.write();
    w.callback_resources.insert(RenderData::new(cc));
    Self { locked_state }
  }
}

struct PaintCallback<GameState> {
  pixel_perfect_size: (u32, u32),
  locked_state:       Arc<Mutex<GameState>>,
}

impl<GameState: PeterEngineApp> CallbackTrait for PaintCallback<GameState> {
  fn prepare(
    &self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    callback_resources: &mut CallbackResources,
  ) -> Vec<wgpu::CommandBuffer> {
    let render_data = callback_resources.get_mut::<RenderData>().unwrap();
    render_data.pixel_perfect_size = self.pixel_perfect_size;
    self.locked_state.lock().unwrap().prepare(device, queue, encoder, render_data)
  }

  fn paint<'rp>(
    &'rp self,
    info: eframe::epaint::PaintCallbackInfo,
    render_pass: &mut wgpu::RenderPass<'rp>,
    callback_resources: &'rp CallbackResources,
  ) {
    let render_data = callback_resources.get::<RenderData>().unwrap();
    self.locked_state.lock().unwrap().paint(info, render_pass, render_data)
  }
}

impl<GameState: PeterEngineApp> eframe::App for EframeApp<GameState> {
  fn update(&mut self, egui_ctx: &egui::Context, frame: &mut eframe::Frame) {
    let dt = egui_ctx.input(|inp| inp.stable_dt.clamp(0.0, 0.15));
    {
      let mut guard = self.locked_state.lock().unwrap();
      guard.draw_main_gui(egui_ctx, frame, dt);
    }

    egui::CentralPanel::default().frame(egui::Frame::none().fill(egui::Color32::BLACK)).show(
      egui_ctx,
      |ui| {
        let (id, allocated_rect) = ui.allocate_space(ui.available_size());
        let response = ui.interact(allocated_rect, id, egui::Sense::click_and_drag());
        self.locked_state.lock().unwrap().central_panel_input(&egui_ctx, response, &allocated_rect);

        let painter = ui.painter();
        // FIXME: Can I somehow get a ScreenDescriptor?
        let pixel_perfect_rect = egui::Rect::from_two_pos(
          painter.round_pos_to_pixels(allocated_rect.left_top()),
          painter.round_pos_to_pixels(allocated_rect.right_bottom()),
        );
        let pixels_per_point = egui_ctx.pixels_per_point();
        // We clamp with 1 to avoid some divide by zeros and wgpu errors.
        let pixel_perfect_size = (
          ((pixel_perfect_rect.width() * pixels_per_point).round() as u32).max(1),
          ((pixel_perfect_rect.height() * pixels_per_point).round() as u32).max(1),
        );

        painter.add(Callback::new_paint_callback(pixel_perfect_rect, PaintCallback {
          pixel_perfect_size,
          locked_state: Arc::clone(&self.locked_state),
        }));
      },
    );
  }
}
