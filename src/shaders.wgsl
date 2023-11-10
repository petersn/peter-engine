// NB: Keep this value in sync with simulation.rs.
const VOLUME_EDGE_LENGTH: i32 = 10;

struct Uniforms {
  transform_m:       mat4x4<f32>,
  transform_vm:      mat4x4<f32>,
  transform_pvm:     mat4x4<f32>,
  transform_pvm_inv: mat4x4<f32>,
  near:              f32,
  far:               f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) uv: vec2<f32>,
}

// struct RayPathVertex {-
//   @location(0) ray_path_position: vec3<f32>,
// }

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) uv: vec2<f32>,
}

struct VoxelInstance {
  @location(3) voxel_position: vec3<f32>,
  @location(4) voxel_tint: vec3<f32>,
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
}

// ==================== vertex shaders ====================

@vertex
fn vertex_shader_main(
  model: VertexInput,
) -> VertexOutput {
  var out: VertexOutput;
  out.color = model.color;
  out.uv = model.uv;
  out.clip_position = uniforms.transform_pvm * vec4<f32>(model.position, 1.0);
  return out;
}

@vertex
fn voxel_vertex_shader_main(
  model: VertexInput,
  instance: VoxelInstance,
) -> VertexOutput {
  var out: VertexOutput;
  out.color = model.color * instance.voxel_tint;
  out.uv = model.uv;
  out.clip_position = uniforms.transform_pvm * vec4<f32>(model.position + instance.voxel_position, 1.0);
  return out;
}

// @vertex
// fn ray_paths_vertex_shader_main(
//   model: RayPathVertex,
// ) -> VertexOutput {
//   var out: VertexOutput;
//   out.color = vec3<f32>(0.3, 1.0, 0.2);
//   out.uv = vec2<f32>(0.0, 0.0);
//   out.clip_position = uniforms.transform_pvm * vec4<f32>(model.ray_path_position, 1.0);
//   return out;
// }

@vertex
fn full_screen_vertex_shader_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  var out: VertexOutput;
  let dest_pos = vec2<f32>(
    f32(vertex_index & 1u),
    f32((vertex_index >> 1u) & 1u),
  );
  out.clip_position = vec4<f32>(
    2.0 * dest_pos.x - 1.0,
    1.0 - 2.0 * dest_pos.y,
    0.0, 1.0,
  );
  out.color = vec3<f32>(1.0, 1.0, 1.0);
  //out.uv = dest_pos;
  out.uv = out.clip_position.xy;
  return out;
}

// ==================== fragment shaders ====================

@group(1) @binding(0) var data_texture: texture_2d<u32>;
@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

@fragment
fn solid_fragment_shader_main(in: VertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}

@fragment
fn edge_highlight_fragment_shader_main(in: VertexOutput) -> @location(0) vec4<f32> {
  //return vec4<f32>(in.uv, 0.0, 1.0);
  if in.uv.x < 0.1 || in.uv.x > 0.9 || in.uv.y < 0.1 || in.uv.y > 0.9 {
    return vec4<f32>(0.2 * in.color, 1.0);
  }
  return vec4<f32>(in.color, 1.0);
}

@fragment
fn shadow_edge_highlight_fragment_shader_main(in: VertexOutput) -> @location(0) vec4<f32> {
    //return vec4<f32>(in.uv, 0.0, 1.0);
  if in.uv.x < 0.1 || in.uv.x > 0.9 || in.uv.y < 0.1 || in.uv.y > 0.9 {
    return vec4<f32>(0.2 * in.color, 0.1);
  }
  return vec4<f32>(in.color, 0.1);
}

@fragment
fn textured_fragment_shader_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let result = textureSample(t_diffuse, s_diffuse, in.uv);
  if result.a < 0.5 {
    discard;
  }
  return result * vec4<f32>(in.color, 1.0);
}

fn dehomog(v: vec4<f32>) -> vec3<f32> {
  return vec3<f32>(v.x / v.w, v.y / v.w, v.z / v.w);
}

@fragment
fn full_screen_fragment_shader_main(in: VertexOutput) -> FragmentOutput {
  var out: FragmentOutput;
  let camera_origin = dehomog(uniforms.transform_pvm_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0));
  let camera_dest = dehomog(uniforms.transform_pvm_inv * vec4<f32>(in.uv, 1.0, 1.0));
  let camera_forward = normalize(camera_dest - camera_origin);

  // Ray cast onto the main AABB.
  let edge_len = f32(VOLUME_EDGE_LENGTH);
  let inverse_dir = 1.0 / camera_forward;
  let hit_times0 = -camera_origin * inverse_dir;
  let hit_times1 = (vec3<f32>(edge_len, edge_len, edge_len) - camera_origin) * inverse_dir;
  let axis_tmin = vec3<f32>(min(hit_times0.x, hit_times1.x), min(hit_times0.y, hit_times1.y), min(hit_times0.z, hit_times1.z));
  let axis_tmax = vec3<f32>(max(hit_times0.x, hit_times1.x), max(hit_times0.y, hit_times1.y), max(hit_times0.z, hit_times1.z));
  // Compute the bounds of our intersection.
  let aabb_tmin = max(axis_tmin.x, max(axis_tmin.y, axis_tmin.z));
  let aabb_tmax = min(axis_tmax.x, min(axis_tmax.y, axis_tmax.z));
  if aabb_tmax < aabb_tmin || aabb_tmax < 0.0 {
    discard;
  }
  let t_intersect = max(aabb_tmin, 0.0);
  let pos = camera_origin + t_intersect * camera_forward;
  let inverse_dir_abs = abs(inverse_dir);
  var cell: vec3<i32> = vec3<i32>(
    max(0, min(i32(pos.x), VOLUME_EDGE_LENGTH - 1)),
    max(0, min(i32(pos.y), VOLUME_EDGE_LENGTH - 1)),
    max(0, min(i32(pos.z), VOLUME_EDGE_LENGTH - 1)),
  );
  let residue = pos - vec3<f32>(cell);
  let step: vec3<i32> = vec3<i32>(
    select(-1, 1, inverse_dir.x > 0.0),
    select(-1, 1, inverse_dir.y > 0.0),
    select(-1, 1, inverse_dir.z > 0.0),
  );
  var tmax: vec3<f32> = vec3<f32>(
    select(residue.x, 1.0 - residue.x, step.x > 0),
    select(residue.y, 1.0 - residue.y, step.y > 0),
    select(residue.z, 1.0 - residue.z, step.z > 0),
  ) * inverse_dir_abs;

  // We might instantly hit a filled voxel.
  // If so, we need to already know which face of the AABB we hit.
  // We determine what face we hit using these values.
  var x_minimum: bool = axis_tmin.x > axis_tmin.y && axis_tmin.x > axis_tmin.z;
  var y_minimum: bool = axis_tmin.y > axis_tmin.x && axis_tmin.y > axis_tmin.z;
  var z_minimum: bool = !(x_minimum || y_minimum);
  var hit_distance = 0.0;

  while
    0 <= cell.x && cell.x < VOLUME_EDGE_LENGTH &&
    0 <= cell.y && cell.y < VOLUME_EDGE_LENGTH &&
    0 <= cell.z && cell.z < VOLUME_EDGE_LENGTH
  {
    let linear_index = u32(cell.x + cell.y * VOLUME_EDGE_LENGTH + cell.z * VOLUME_EDGE_LENGTH * VOLUME_EDGE_LENGTH);
    let x = linear_index & 0x3ffu;
    let y = linear_index >> 10u;
    let cell_contents = textureLoad(data_texture, vec2<i32>(i32(x), i32(y)), 0).x;
    if cell_contents != 0u {
      // FIXME: This depth computation is somehow a bit broken.
      let hit_point = pos + (t_intersect + hit_distance) * camera_forward;
      let eye_space = uniforms.transform_pvm * vec4<f32>(hit_point, 1.0);
      out.depth = eye_space.z / eye_space.w;
      //let depth = (t_intersect + hit_distance) * dot_product(local_ray_direction, local_camera_forward);
      if x_minimum {
        if step.x > 0 {
          out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
        } else {
          out.color = vec4<f32>(0.0, 1.0, 0.0, 1.0);
        }
      }
      if y_minimum {
        if step.y > 0 {
          out.color = vec4<f32>(0.0, 0.0, 1.0, 1.0);
        } else {
          out.color = vec4<f32>(1.0, 1.0, 0.0, 1.0);
        }
      }
      if z_minimum {
        if step.z > 0 {
          out.color = vec4<f32>(1.0, 0.0, 1.0, 1.0);
        } else {
          out.color = vec4<f32>(0.0, 1.0, 1.0, 1.0);
        }
      }
      return out;
    }

    x_minimum = tmax.x < tmax.y && tmax.x < tmax.z;
    y_minimum = tmax.y < tmax.x && tmax.y < tmax.z;
    z_minimum = !(x_minimum || y_minimum);
    hit_distance = min(tmax.x, min(tmax.y, tmax.z));
    tmax.x += select(0.0, inverse_dir_abs.x, x_minimum);
    tmax.y += select(0.0, inverse_dir_abs.y, y_minimum);
    tmax.z += select(0.0, inverse_dir_abs.z, z_minimum);
    cell.x += select(0, step.x, x_minimum);
    cell.y += select(0, step.y, y_minimum);
    cell.z += select(0, step.z, z_minimum);
  }

  discard;
  //return vec4<f32>(1.0, 0.0, 0.0, 1.0);

  /*

  //let x = u32(in.uv.x * 1024.0);
  //let y = u32(in.uv.y * 1024.0);
  let texture_index: u32 = x + (y << 10u);
  let data: u32 = textureLoad(data_texture, vec2<i32>(i32(texture_index & 0x3ffu), i32(texture_index >> 10u)), 0).x;
  let r = data & 0xffu;
  let g = (data >> 8u) & 0xffu;
  let b = (data >> 16u) & 0xffu;
  return vec4<f32>(f32(r), f32(g), f32(b), 255.0) / 255.0;
  */
}
