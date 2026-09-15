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
| `image/results/<scene>_pbr.webp` | methods down the rows, modalities across; column 0 is each method's own preprocessed input, exposed to a common foreground mean so the column is comparable &mdash; ours is solved in the linear HDR domain before the gamma, and estimated maps are untouched. Fixed source height (888 px), width follows the object; the page pins one displayed height for every scene and lets the figure run wider than the text column so the flattest object still fits |
| `image/results/<scene>_relight.webp` | `PDIR/outputs/render_12pat_5method_bright_crop`, pattern `AntiDiag` |
| `image/envgrid/o{1..3}_l{1..3}.webp` | 12 objects x 20 environments as twelve 4x5 grids, object set and lighting set chosen independently on the page. `fg_gain=2.7` (Shanghai Bund at exposure 0.5, it tone-maps far brighter than the rest), background through `make_perspective_bg` at 65 deg rather than the module's default panoramic crop, mirror ball top-right from `render_env_3scene_5method_xflip/_chromeballs` |
| `video/dynamic_{capture,relit}.mp4` | `outputs/face_pipeline_scene4_20260914_vanila/_videos_realtime_capturefps`, re-encoded from MPEG-4 to H.264 so browsers can play them |
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

The environment renderer backs the object with `make_panoramic_bg`, which spans
a full 360 deg of longitude across the image width while covering 90 deg of
latitude down its height, so a square frame reads as horizontally squeezed. The
grids swap in `make_perspective_bg`, a pinhole projection, at a 65 deg vertical
field of view with the pitch taken from `bg_v_center`.

`static/js/relight.js` reproduces `Principled_BRDF.forward`: the light circles
at radius 0.4 about (0, -0.29, 0) while the shaded point sits at (0, 0, 0.5),
the incident direction has its y and z flipped, and the diffuse lobe carries the
Disney retro-reflection term. The shaded values were checked against
`render_relight.py` on scene29 at four angles: max 4/255, mean 0.01/255. The
tone map matches too &mdash; clip to [0,1] at gain 50, no gamma encoding, the
same as `--brightness 50`. A dim camera-side fill (0.2) is added on top so the
shadowed side stays readable; that part is not in the paper.

Table numbers are transcribed from the paper (Tables 1 and 2 in `7_results.tex`).
Column headers carry `data-dir="higher"`/`"lower"`, and `static/js/index.js`
marks the best and second-best value in each column from that.
