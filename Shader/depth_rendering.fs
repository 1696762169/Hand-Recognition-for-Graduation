#version 430 core

#if USE_BINDLESS_TEXTURE
#extension GL_ARB_bindless_texture : require
#endif
#extension GL_EXT_texture_array : require

in GeometryOut
{
    mat4 affine_transform;
    vec3 view_pos;
    mat3 view_TBN;
    vec3 tex_coord;
    vec4 color;
    vec4 back_color;
    flat int visible;
} fs_in;

#if USE_BINDLESS_TEXTURE && USE_DYNAMIC_ENV_MAPPING
in flat uvec2 env_map_handle;
#endif

layout(location=0) out vec4 out_color;
layout(location=1) out vec4 accum;
layout(location=2) out float reveal;
layout(location=3) out vec3 view_pos;
layout(location=4) out vec3 view_normal;

uniform bool is_opaque_pass;
uniform bool is_sphere;
uniform vec3 mesh_center;

uniform float depth_min;
uniform float depth_max;

uniform bool background;    // 无用变量

void main()
{
    // 将深度作为颜色渲染
    float depth = (fs_in.view_pos.y - depth_min) / (depth_max - depth_min);
    depth = 1.0 - clamp(depth, 0.0, 1.0);
    out_color = vec4(depth, depth, depth, 1.0);
}