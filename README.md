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
rasterized from `figures/*.pdf` in the ACM submission bundle with PyMuPDF.

Everything else is built from inference output. Every local asset URL carries a
`?v=N` tag, set in `index.html` and in `ASSET_V` in both scripts; bump it
whenever an asset changes, or browsers and the Pages CDN keep serving the old
file under the same name.

| Asset | Built from |
|---|---|
| `image/results/<scene>_pbr.webp` | modalities down the rows, methods across; row 0 is each method's own preprocessed input, shown with a display-only exposure (ours x0.78, baselines x1.5) so the columns are comparable &mdash; estimated maps are untouched. Fixed source height (1102 px), width follows the object; the page pins the displayed height so every scene renders the same size |
| `image/results/<scene>_relight.webp` | `PDIR/outputs/render_12pat_5method_bright_crop`, pattern `AntiDiag` |
| `image/envgrid/page*.webp` | 13 objects x 16 environments, `fg_gain=2.7`, mirror ball top-right from `render_env_3scene_5method_xflip/_chromeballs` |
| `webgl/<scene>_{albedo,normal,mat}.png` | `pipeline_vanila_valid_final` maps; `mat` packs roughness, metallicity and mask into R, G, B |

Per-method sources (`<safe>` = `<prefix>__<scene>`):

```
Ours    input  pattern_main_hdr_rgb.npy (the raw snapshot)
        maps   SIGGA2026/PDIR/outputs/pipeline_vanila_valid_final/<scene>/
LINO    input  SIGGA2026/lino_pbr/outputs/pdir_real_avg/<safe>/input_avg.png
        maps   same directory, basecolor.png for albedo
DR      input  SIGGA2026/diffusion-renderer/input_real_valid_white_masked/<safe>/frame_00000.png
        maps   SIGGA2026/diffusion-renderer/output_real_valid_white_masked/<safe>/0000.0000.*
RGB-X   input  SIGGA2026/rgbx/rgbx_dataset/input_real_valid_white/<safe>/frame_00000.png
        maps   SIGGA2026/rgbx/rgbx_dataset/output_rgbx_real_valid_white/<safe>/
```

`static/js/relight.js` reproduces `Principled_BRDF.forward`: the light circles
at radius 0.4 about (0, -0.29, 0) while the shaded point sits at (0, 0, 0.5),
the incident direction has its y and z flipped, and the diffuse lobe carries the
Disney retro-reflection term. Checked against `render_relight.py` on scene29 at
four angles: max 4/255, mean 0.01/255. A dim camera-side fill (0.2) is added on
top so the shadowed side stays readable; that part is not in the paper.

Table numbers are transcribed from the paper (Tables 1 and 2 in `7_results.tex`).
Column headers carry `data-dir="higher"`/`"lower"`, and `static/js/index.js`
marks the best and second-best value in each column from that.
