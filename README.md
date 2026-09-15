# PDIR — project page

Project page for **Snapshot Polarimetric Display Inverse Rendering**
(ACM Transactions on Graphics 45(6), Article 201 — Proc. SIGGRAPH Asia 2026).

This branch (`project_page`) holds only the website. The training and inference
code lives on `main`.

## Layout

```
index.html                         the whole page
static/css/bulma.min.css           Bulma (vendored)
static/css/index.css               NeRFies base styles + page-specific styles
static/js/index.js                 table best/second-best marking, no dependencies
static/js/fontawesome.all.min.js   icons (vendored)
static/image/*.webp                paper figures
```

## Local preview

```bash
python -m http.server 8000
# then open http://localhost:8000
```

## Publishing

Settings → Pages → *Deploy from a branch* → branch `project_page`, folder `/ (root)`.
`.nojekyll` is present so GitHub serves the files as-is.

## Figures and interactive parts

Paper figures (teaser, imaging system, network overview, pBRDF expansion) are
rasterized from `figures/*.pdf` in the ACM submission bundle with PyMuPDF at
about twice their display width and saved as WebP.

Everything in Results, Move the light and Environment lighting is built from
inference output. Demo scenes are `scene29` (cat), `scene41` (bowl), `scene152`
(aluminium case) and `scene1` (owl).

| Asset | Built from |
|---|---|
| `image/results/<scene>_input.webp` | `pattern_main_hdr_rgb.npy`, tone-mapped at the 99th percentile |
| `image/results/<scene>_pbr.webp` | five methods x four maps; source paths below |
| `image/results/<scene>_relight.webp` | `PDIR/outputs/render_12pat_5method_bright_crop`, pattern `AntiDiag` |
| `image/envgrid/page*.webp` | 16 objects x 5 environments rendered with `PDIR/scripts/make_videos_envmap.py` helpers at `fg_gain=2.7`, mirror balls from `render_env_3scene_5method_xflip/_chromeballs` |
| `webgl/<scene>_{albedo,normal,mat}.png` | `inference.py` output; `mat` packs roughness, metallicity and mask into R, G, B |

Per-method PBR maps for the comparison panel (the same paths
`lino_pbr/build_comparison_grid.py` uses, with `<safe>` = `<prefix>__<scene>`):

```
Ours      SIGGA2026/PDIR/outputs/pipeline_vanila_valid_final/<scene>/{normal,albedo,roughness,metallic}.png
LINO N=4  SIGGA2026/lino_pbr/outputs/pdir_real_four/<safe>/{normal,basecolor,roughness,metallic}.png
LINO N=1  SIGGA2026/lino_pbr/outputs/pdir_real_avg/<safe>/...
DR        SIGGA2026/diffusion-renderer/output_real_valid_white_masked/<safe>/0000.0000.{normal,basecolor,roughness,metallic}.png
RGB-X     SIGGA2026/rgbx/rgbx_dataset/output_rgbx_real_valid_white/<safe>/{normal,albedo,roughness,metallic}.png
```

`static/js/relight.js` shades the WebGL textures with the paper's principled
BRDF (GGX + Smith + Schlick, `F0 = 0.08(1-m) + albedo*m`), lit by one orbiting
key light at a fixed elevation plus a dim camera-side fill. Only the azimuth is
exposed to the reader; the stage is capped at 340 px so the 384 px maps are
never shown much above native resolution.

Table numbers are transcribed from the paper (Tables 1 and 2 in `7_results.tex`).
Column headers carry `data-dir="higher"`/`"lower"`, and `static/js/index.js`
marks the best and second-best value in each column from that.
