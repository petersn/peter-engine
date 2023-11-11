use eframe::WebRunner;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  pub fn log(s: &str);
}

#[derive(Clone)]
#[wasm_bindgen]
pub struct WebHandle {
  runner: WebRunner,
}

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
