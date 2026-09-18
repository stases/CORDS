# CORDS identity

`cords-banner.svg` is the README banner: a set of objects on an upper plane,
their drop lines, and below them the density field ρ(r) as iso-lines with the
per-class feature field h(r) as colour. The field is a computed kernel sum
ρ(r) = Σᵢ κ_σ(r − rᵢ); the contours come from the contour engine, not from a
drawing tool. The type is Source Sans 3 converted to outlines, so the SVG
renders identically everywhere without fonts. The card has its own dark ground
with transparent rounded corners, so it sits on both GitHub themes.

| File | Use |
| --- | --- |
| `cords-banner.svg`, `cords-banner@2x.png` | README header (1600×500, rounded card) |
| `cords-banner-B-planes.svg` | Same scene with a one-line tagline, for slides |
| `cords-mark-B-planes.svg`, `cords-mark-B-planes@2x.png` | Square mark (512×512) for avatar, favicon, slide corner |

Object hues sit at one lightness and chroma in oklch so no class dominates;
ground `#12171d`, type `#f2eee6`.

Regenerate with

```bash
/home/thadziv/miniconda3/bin/python docs/branding/make_branding.py docs/branding
```

The script also emits two alternative directions (a kernel-in-the-O wordmark
and a stem-plot banner) that are not used in the README.
