/* In-browser relighting of the estimated PBR maps.
   A fragment shader evaluates the same principled BRDF the paper renders with:
   Lambert diffuse scaled by (1 - metallicity), GGX specular with Smith masking
   and a Schlick Fresnel whose F0 is 0.08 blended toward the albedo by
   metallicity. Textures are albedo (RGB), normal ([0,1] encoded) and a packed
   map holding roughness, metallicity and the object mask. */

(function () {
  'use strict';

  var ASSET_V = '?v=3';

  var SCENES = [
    { label: 'Cat', key: 'scene29' },
    { label: 'Bowl', key: 'scene41' },
    { label: 'Case', key: 'scene152' },
    { label: 'Owl', key: 'scene1' }
  ];

  // The light stays on one orbit; only its azimuth is exposed to the reader.
  var ELEVATION_DEG = 25;
  var GAIN = 2.6;
  var FILL = 0.3;

  var VERT = [
    'attribute vec2 a_pos;',
    'varying vec2 v_uv;',
    'void main() {',
    '  v_uv = vec2(a_pos.x * 0.5 + 0.5, 0.5 - a_pos.y * 0.5);',
    '  gl_Position = vec4(a_pos, 0.0, 1.0);',
    '}'
  ].join('\n');

  var FRAG = [
    'precision highp float;',
    'varying vec2 v_uv;',
    'uniform sampler2D u_albedo;',
    'uniform sampler2D u_normal;',
    'uniform sampler2D u_mat;',
    'uniform vec3 u_light;',
    'uniform float u_gain;',
    'uniform float u_fill;',
    'const float PI = 3.14159265359;',
    '',
    'float ggx(float ndoth, float a) {',
    '  float a2 = a * a;',
    '  float d = ndoth * ndoth * (a2 - 1.0) + 1.0;',
    '  return a2 / max(PI * d * d, 1e-7);',
    '}',
    'float smithG(float ndotx, float a) {',
    '  float a2 = a * a;',
    '  return 1.0 / max(ndotx + sqrt(a2 + (1.0 - a2) * ndotx * ndotx), 1e-7);',
    '}',
    '',
    'vec3 shade(vec3 l, vec3 n, vec3 albedo, float rough, float metal) {',
    '  vec3 v = vec3(0.0, 0.0, 1.0);',
    '  vec3 h = normalize(l + v);',
    '  float ndotl = max(dot(n, l), 0.0);',
    '  float ndotv = max(dot(n, v), 1e-4);',
    '  float ndoth = max(dot(n, h), 0.0);',
    '  float ldoth = max(dot(l, h), 0.0);',
    '  float a = rough * rough;',
    '  vec3 f0 = vec3(0.08) * (1.0 - metal) + albedo * metal;',
    '  vec3 fres = f0 + (1.0 - f0) * pow(1.0 - ldoth, 5.0);',
    '  vec3 spec = fres * ggx(ndoth, a) * smithG(ndotl, a) * smithG(ndotv, a);',
    '  vec3 diff = (1.0 - metal) * albedo / PI;',
    '  return (diff + spec) * ndotl;',
    '}',
    '',
    'void main() {',
    '  vec3 mat = texture2D(u_mat, v_uv).rgb;',
    '  float mask = mat.b;',
    '  if (mask < 0.5) { gl_FragColor = vec4(0.055, 0.06, 0.07, 1.0); return; }',
    '',
    '  vec3 albedo = texture2D(u_albedo, v_uv).rgb;',
    '  vec3 n = normalize(texture2D(u_normal, v_uv).rgb * 2.0 - 1.0);',
    '  float rough = clamp(mat.r, 0.05, 1.0);',
    '  float metal = clamp(mat.g, 0.0, 1.0);',
    '',
    '  // key light on the orbit, plus a dim fill from the camera so the',
    '  // shadowed side stays readable instead of clipping to black',
    '  vec3 color = shade(normalize(u_light), n, albedo, rough, metal);',
    '  color += u_fill * shade(vec3(0.0, 0.0, 1.0), n, albedo, rough, metal);',
    '  color *= u_gain;',
    '  color = pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));',
    '  gl_FragColor = vec4(color, 1.0);',
    '}'
  ].join('\n');

  function compile(gl, type, source) {
    var s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s));
    }
    return s;
  }

  function makeTexture(gl, unit) {
    var tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // 1x1 placeholder so the first draw before load does not error
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                  new Uint8Array([0, 0, 0, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    return tex;
  }

  function init() {
    var canvas = document.getElementById('relightCanvas');
    var fallback = document.getElementById('relightFallback');
    if (!canvas) return;

    var gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
      canvas.hidden = true;
      if (fallback) fallback.hidden = false;
      return;
    }

    var program = gl.createProgram();
    try {
      gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERT));
      gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program));
      }
    } catch (err) {
      console.error('relight shader:', err);
      canvas.hidden = true;
      if (fallback) fallback.hidden = false;
      return;
    }
    gl.useProgram(program);

    var buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER,
                  new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    var aPos = gl.getAttribLocation(program, 'a_pos');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    var units = { albedo: 0, normal: 1, mat: 2 };
    var textures = {};
    Object.keys(units).forEach(function (name) {
      textures[name] = makeTexture(gl, units[name]);
      gl.uniform1i(gl.getUniformLocation(program, 'u_' + name), units[name]);
    });

    var uLight = gl.getUniformLocation(program, 'u_light');
    var uGain = gl.getUniformLocation(program, 'u_gain');
    var uFill = gl.getUniformLocation(program, 'u_fill');

    var angleEl = document.getElementById('relightAngle');
    if (!angleEl) {                       // markup and script out of step
      console.error('relight: #relightAngle missing');
      if (fallback) fallback.hidden = false;
      return;
    }
    var el = ELEVATION_DEG * Math.PI / 180;
    var cosEl = Math.cos(el);
    var sinEl = Math.max(Math.sin(el), 0.05);

    function draw() {
      var az = (parseFloat(angleEl.value) || 0) * Math.PI / 180;
      gl.uniform3f(uLight, Math.cos(az) * cosEl, Math.sin(az) * cosEl, sinEl);
      gl.uniform1f(uGain, GAIN);
      gl.uniform1f(uFill, FILL);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    function loadScene(key) {
      Object.keys(units).forEach(function (name) {
        var img = new Image();
        img.onload = function () {
          gl.activeTexture(gl.TEXTURE0 + units[name]);
          gl.bindTexture(gl.TEXTURE_2D, textures[name]);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
          draw();
        };
        img.src = './static/webgl/' + key + '_' + name + '.png' + ASSET_V;
      });
    }

    /* scene buttons */
    var tabs = document.getElementById('relightTabs');
    var buttons = SCENES.map(function (scene, i) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'scene-tab';
      b.textContent = scene.label;
      b.addEventListener('click', function () { select(i); });
      tabs.appendChild(b);
      return b;
    });
    function select(i) {
      buttons.forEach(function (b, k) { b.classList.toggle('is-active', k === i); });
      loadScene(SCENES[i].key);
    }

    angleEl.addEventListener('input', draw);

    /* dragging the canvas moves the light */
    var dragging = false;
    function pointerAngle(ev) {
      var rect = canvas.getBoundingClientRect();
      var touch = ev.touches && ev.touches[0];
      var x = (touch ? touch.clientX : ev.clientX) - rect.left - rect.width / 2;
      var y = rect.height / 2 - ((touch ? touch.clientY : ev.clientY) - rect.top);
      var deg = Math.atan2(y, x) * 180 / Math.PI;
      angleEl.value = ((deg % 360) + 360) % 360;
      draw();
    }
    canvas.addEventListener('mousedown', function (e) { dragging = true; pointerAngle(e); });
    window.addEventListener('mousemove', function (e) { if (dragging) pointerAngle(e); });
    window.addEventListener('mouseup', function () { dragging = false; });
    canvas.addEventListener('touchstart', function (e) { e.preventDefault(); pointerAngle(e); }, { passive: false });
    canvas.addEventListener('touchmove', function (e) { e.preventDefault(); pointerAngle(e); }, { passive: false });

    select(0);
    draw();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
