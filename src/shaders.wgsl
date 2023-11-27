struct CameraUniforms {
  transform_m:       mat4x4<f32>,
  transform_vm:      mat4x4<f32>,
  transform_pvm:     mat4x4<f32>,
  transform_pvm_inv: mat4x4<f32>,
  near:              f32,
  far:               f32,
}

@group(0) @binding(0) var<uniform> camera_uniforms: CameraUniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) color: vec4<f32>,
  @location(2) uv: vec2<f32>,
}

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) screen_space_uv: vec2<f32>,
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
}

// // ==================== vertex shaders ====================

// @vertex
// fn default_vertex_shader(
//   model: VertexInput,
// ) -> VertexOutput {
//   var out: VertexOutput;
//   out.color = model.color;
//   out.uv = model.uv;
//   out.clip_position = uniforms.transform_pvm * vec4<f32>(model.position, 1.0);
//   return out;
// }

// @vertex
// fn full_screen_vertex_shader(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
//   var out: VertexOutput;
//   let dest_pos = vec2<f32>(
//     f32(vertex_index & 1u),
//     f32((vertex_index >> 1u) & 1u),
//   );
//   out.clip_position = vec4<f32>(
//     2.0 * dest_pos.x - 1.0,
//     1.0 - 2.0 * dest_pos.y,
//     0.0, 1.0,
//   );
//   out.color = vec3<f32>(1.0, 1.0, 1.0);
//   out.uv = out.clip_position.xy;
//   return out;
// }

// // ==================== fragment shaders ====================

// @group(1) @binding(0) var data_texture: texture_2d<u32>;
// @group(1) @binding(0) var t_diffuse: texture_2d<f32>;
// @group(1) @binding(1) var s_diffuse: sampler;

// @fragment
// fn solid_fragment_shader(in: VertexOutput) -> @location(0) vec4<f32> {
//   return vec4<f32>(in.color, 1.0);
// }

// @fragment
// fn textured_fragment_shader(in: VertexOutput) -> @location(0) vec4<f32> {
//   let result = textureSample(t_diffuse, s_diffuse, in.uv);
//   return result * vec4<f32>(in.color, 1.0);
// }
