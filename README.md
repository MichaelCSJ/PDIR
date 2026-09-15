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
about twice their display width and saved as WebP (quality 88).

The results, environment-lighting and relighting sections are built from
inference outputs rather than the paper:

| Asset | Built from |
|---|---|
| `static/image/results/<scene>_pbr.webp` | `inference.py` output for the four demo scenes |
| `static/image/results/<scene>_relight.webp` | `outputs/render_12pat_5method_bright_crop`, pattern `AntiDiag` |
| `static/image/envgrid/page*.webp` | `outputs/render_env_3scene_5method_xflip`, with mirror balls from its `_chromeballs` |
| `static/webgl/<scene>_{albedo,normal,mat}.png` | the same `maps.npz`; `mat` packs roughness, metallicity and mask into R, G, B |

`static/js/relight.js` shades those textures live in WebGL with the paper's
principled BRDF (GGX + Smith + Schlick, `F0 = 0.08(1-m) + albedo*m`), lit by one
orbiting key light plus a dim camera-side fill so the shadowed side stays
readable.

Table numbers are transcribed from the paper (Tables 1 and 2 in `7_results.tex`).
Column headers carry `data-dir="higher"`/`"lower"`, and `static/js/index.js`
marks the best and second-best value in each column from that.
