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

## Figures

Every figure is the camera-ready figure from the paper, rasterized from
`figures/*.pdf` in the ACM submission bundle (`tog456-article201.zip`) at about
twice its display width and saved as WebP (quality 88). To regenerate after a
figure changes:

```python
import pymupdf, io
from PIL import Image

doc = pymupdf.open("figures/teaser.pdf")
page = doc[0]
zoom = 2000 / page.rect.width          # target pixel width
pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB") \
     .save("static/image/teaser.webp", "WEBP", quality=88, method=6)
```

Target widths in use: teaser and dynamic_face 2000, overview 1900,
imaging_system / pbrdf_expansion / ablation 1800, albedo / metallic /
rotating_light / env_map 1600, comparison / expand_data 1500.

Table numbers are transcribed from the paper (Tables 1–4 in `7_results.tex`).
Column headers carry `data-dir="higher"`/`"lower"`, and `static/js/index.js`
marks the best and second-best value in each column from that.
