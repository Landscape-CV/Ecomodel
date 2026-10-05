// Point shader shared by the region/single-cloud viewer and the streaming octree layer.
import * as THREE from "three";

export const PAL_W = 1024;

export const VERT = /* glsl */ `
uniform float uSize;
uniform float uPixelRatio;
uniform float uAtten;
uniform float uScale;
uniform float uMode;        // 0 instance, 1 material, 2 height
uniform vec2 uHRange;       // z range for height shading
uniform float uFocus;
uniform float uFocusOn;
uniform float uIsolate;
uniform float uHideNT;
uniform float uZmin;
uniform float uZmax;
uniform float uShowWood;
uniform float uShowLeaf;
uniform float uSpacing;     // >0: adaptive size from octree node spacing (m)
uniform float uDim;         // context rendering around an edit region
uniform float uClipOn;      // hide points inside the xy box (region shown separately)
uniform vec2 uClipMin;
uniform vec2 uClipMax;
uniform sampler2D uPalette;
uniform vec2 uPalSize;
attribute float label;
attribute float selected;
attribute float material;
varying vec3 vColor;

vec3 treeColor(float id) {
  if (id < 0.0) return vec3(0.42, 0.42, 0.45);
  float id2 = mod(id, uPalSize.x * uPalSize.y);
  vec2 uv = vec2((mod(id2, uPalSize.x) + 0.5) / uPalSize.x, (floor(id2 / uPalSize.x) + 0.5) / uPalSize.y);
  return texture2D(uPalette, uv).rgb;
}

vec3 heightRamp(float t) {
  vec3 a = vec3(0.16, 0.27, 0.62), b = vec3(0.18, 0.62, 0.42), c = vec3(0.96, 0.85, 0.32);
  return t < 0.5 ? mix(a, b, t * 2.0) : mix(b, c, t * 2.0 - 1.0);
}

void main() {
  bool isFocus = uFocusOn > 0.5 && abs(label - uFocus) < 0.5;
  bool clipped = uClipOn > 0.5 && position.x >= uClipMin.x && position.x <= uClipMax.x
    && position.y >= uClipMin.y && position.y <= uClipMax.y;
  bool hide = clipped
    || (uHideNT > 0.5 && label < 0.0)
    || (uIsolate > 0.5 && uFocusOn > 0.5 && !isFocus)
    || position.z < uZmin || position.z > uZmax
    || (material > 0.5 && material < 1.5 && uShowWood < 0.5)
    || (material > 1.5 && uShowLeaf < 0.5);
  if (hide) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    return;
  }
  vec3 col;
  float hn = clamp((position.z - uHRange.x) / max(uHRange.y - uHRange.x, 1e-3), 0.0, 1.0);
  if (uMode > 1.5) {
    col = heightRamp(hn);
  } else if (uMode > 0.5) {
    col = material > 1.5 ? vec3(0.18, 0.55, 0.34) : (material > 0.5 ? vec3(0.55, 0.35, 0.17) : vec3(0.45));
  } else {
    col = label < 0.0 ? vec3(0.24 + 0.42 * hn) : treeColor(label);
  }
  if (uFocusOn > 0.5 && uIsolate < 0.5 && uMode < 0.5 && !isFocus) col = mix(col, vec3(0.12), 0.65);
  if (uDim > 0.5) col = mix(col, vec3(0.16), 0.6);
  if (selected > 0.5) col = vec3(1.0, 0.25, 0.85);
  vColor = col;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float sz = uSize * uPixelRatio;
  if (uAtten > 0.5) sz = uSize * uScale / max(-mv.z, 0.01);
  if (uSpacing > 0.0) sz = clamp(uSpacing * uScale * uPixelRatio / max(-mv.z, 0.01), sz, sz * 4.0);
  if (selected > 0.5) sz = max(sz, 2.0 * uPixelRatio) * 1.15;
  gl_PointSize = clamp(sz, 1.0, 64.0);
}
`;

export const FRAG = /* glsl */ `
uniform float uOpacity;
varying vec3 vColor;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(vColor, uOpacity);
}
`;

export function extraUniforms() {
  return {
    uSpacing: { value: 0 }, uDim: { value: 0 }, uClipOn: { value: 0 },
    uClipMin: { value: new THREE.Vector2() }, uClipMax: { value: new THREE.Vector2() },
  };
}
