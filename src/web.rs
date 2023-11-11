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
  #[wasm_bindgen(skip)]
  pub runner: WebRunner,
}

pub fn clear_loading_message() {
  web_sys::window()
    .unwrap()
    .document()
    .unwrap()
    .get_element_by_id("loadingMessage")
    .unwrap()
    .set_attribute("style", "display: none")
    .unwrap();
  log("Launching app from WASM");
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
