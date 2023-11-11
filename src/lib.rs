// Re-export our main dependencies.
use std::sync::{Arc, Mutex};

pub use eframe;
use eframe::egui_wgpu::{Callback, CallbackResources, CallbackTrait};
pub use eframe::{egui, wgpu};
use graphics::RenderData;
pub use image;
pub use nalgebra;

pub mod graphics;
pub mod mipmapping;
#[cfg(target_arch = "wasm32")]
pub mod web;

#[cfg(not(target_arch = "wasm32"))]
pub fn log(s: &str) {
  println!("{}", s);
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  pub fn log(s: &str);
}

pub trait PeterEngineApp: Send + 'static {
  const WINDOW_TITLE: &'static str;

  type RenderResources: Send + Sync + 'static;

  fn get_shader_source() -> String;
  fn init(
    &mut self,
    cc: &eframe::CreationContext,
    render_data: &mut RenderData,
  ) -> Self::RenderResources;
  fn update(&mut self, egui_ctx: &egui::Context, frame: &mut eframe::Frame, dt: f32);
  fn central_panel_input(
    &mut self,
    _egui_ctx: &egui::Context,
    _response: egui::Response,
    _allocated_rect: &egui::Rect,
  ) {
  }
  fn prepare(
    &mut self,
    render_data: &mut RenderData,
    resources: &mut Self::RenderResources,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    screen_size: (u32, u32),
  ) -> Vec<wgpu::CommandBuffer>;
  fn paint<'rp>(
    &mut self,
    render_data: &'rp RenderData,
    resources: &'rp Self::RenderResources,
    info: eframe::epaint::PaintCallbackInfo,
    render_pass: &mut wgpu::RenderPass<'rp>,
  );
}

pub struct EframeApp<GameState> {
  locked_state: Arc<Mutex<GameState>>,
}

impl<GameState: PeterEngineApp> EframeApp<GameState> {
  pub fn new(mut game_state: GameState, cc: &eframe::CreationContext) -> Self {
    let wgpu_render_state = cc.wgpu_render_state.as_ref().unwrap();
    let mut w = wgpu_render_state.renderer.write();
    let shader_source = GameState::get_shader_source();
    let mut rd = RenderData::new(cc, &shader_source);
    let resources = game_state.init(cc, &mut rd);
    let locked_state = Arc::new(Mutex::new(game_state));
    w.callback_resources.insert((rd, resources));
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
    let (render_data, resources) =
      callback_resources.get_mut::<(RenderData, GameState::RenderResources)>().unwrap();
    render_data.pixel_perfect_size = self.pixel_perfect_size;
    self.locked_state.lock().unwrap().prepare(
      render_data,
      resources,
      device,
      queue,
      encoder,
      self.pixel_perfect_size,
    )
  }

  fn paint<'rp>(
    &self,
    info: eframe::epaint::PaintCallbackInfo,
    render_pass: &mut wgpu::RenderPass<'rp>,
    callback_resources: &'rp CallbackResources,
  ) {
    let (render_data, resources) =
      callback_resources.get::<(RenderData, GameState::RenderResources)>().unwrap();
    self.locked_state.lock().unwrap().paint(render_data, resources, info, render_pass)
  }
}

impl<GameState: PeterEngineApp> eframe::App for EframeApp<GameState> {
  fn update(&mut self, egui_ctx: &egui::Context, frame: &mut eframe::Frame) {
    let dt = egui_ctx.input(|inp| inp.stable_dt.clamp(0.0, 0.15));
    {
      let mut guard = self.locked_state.lock().unwrap();
      guard.update(egui_ctx, frame, dt);
    }

    egui::CentralPanel::default().frame(egui::Frame::none().fill(egui::Color32::BLACK)).show(
      egui_ctx,
      |ui| {
        let (id, allocated_rect) = ui.allocate_space(ui.available_size());
        let response = ui.interact(allocated_rect, id, egui::Sense::click_and_drag());
        self
          .locked_state
          .lock()
          .unwrap()
          .central_panel_input(&egui_ctx, response, &allocated_rect);

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

#[cfg(not(target_arch = "wasm32"))]
pub fn launch<GameState: PeterEngineApp>(game_state: GameState) -> Result<(), eframe::Error> {
  let mut native_options = eframe::NativeOptions::default();
  native_options.depth_buffer = 32;
  native_options.multisampling = crate::graphics::MSAA_COUNT as u16;
  eframe::run_native(
    GameState::WINDOW_TITLE,
    native_options,
    Box::new(move |cc| Box::new(EframeApp::new(game_state, cc))),
  )
}
