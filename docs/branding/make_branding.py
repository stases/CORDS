"""Generate CORDS logo marks and README banners as self-contained SVG.

Three directions, each with a 1600x500 banner and a 512x512 mark:

  A  kernel   light, editorial: the counter of the O in CORDS holds the
              iso-lines of a single Gaussian kernel around its centre point.
  B  planes   dark, after Figure 1 of the paper: a set of objects on an
              upper plane, drop lines, and the density/feature field below.
  C  comb     light, one-dimensional: a stem plot of unit masses and the
              kernel sum that envelops it, sharing a baseline with the type.

All fields are true kernel sums rho(r) = sum_i kappa_sigma(r - r_i); the
contour lines are computed, not drawn by hand. Text is converted to glyph
outlines from Source Sans 3 so the SVGs need no fonts installed.

Run:  /home/thadziv/miniconda3/bin/python make_branding.py [outdir]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FONT_DIR = Path("/home/thadziv/GitHub/design/assets/fonts")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)

TAGLINE = "CONTINUOUS REPRESENTATIONS OF DISCRETE STRUCTURES"


# --------------------------------------------------------------------------
# colour: oklch -> sRGB hex, so every accent shares lightness and chroma
# --------------------------------------------------------------------------
def oklch(L: float, C: float, h_deg: float) -> str:
    h = math.radians(h_deg)
    a, b = C * math.cos(h), C * math.sin(h)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    def gam(c):
        c = min(max(c, 0.0), 1.0)
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

    return "#%02x%02x%02x" % tuple(round(gam(c) * 255) for c in (r, g, bb))


INK = "#172029"          # cool near-black
PAPER = "#f6f4ee"        # warm off-white
PAPER_DARK = "#12171d"   # banner B ground
COPPER = oklch(0.62, 0.15, 45)     # direction A accent
TEAL = oklch(0.62, 0.15, 225)      # direction C accent
# direction B: object classes, all at one lightness/chroma, hue rotates
CLASS_HUES = [45, 95, 150, 200, 250, 300, 340]
CLASS = [oklch(0.76, 0.14, h) for h in CLASS_HUES]


# --------------------------------------------------------------------------
# type: glyph outlines with GPOS pair kerning
# --------------------------------------------------------------------------
class Face:
    def __init__(self, path: Path):
        self.font = TTFont(path)
        self.upem = self.font["head"].unitsPerEm
        self.cmap = self.font.getBestCmap()
        self.glyphs = self.font.getGlyphSet()
        os2 = self.font["OS/2"]
        self.cap_height = getattr(os2, "sCapHeight", 0.66 * self.upem)
        self.kern = self._load_kerning()

    def _load_kerning(self):
        pairs = {}
        if "GPOS" not in self.font:
            return pairs
        gpos = self.font["GPOS"].table
        for lookup in gpos.LookupList.Lookup:
            subtables = lookup.SubTable
            if lookup.LookupType == 9:  # extension
                subtables = [st.ExtSubTable for st in subtables]
            for st in subtables:
                if getattr(st, "LookupType", lookup.LookupType) != 2:
                    continue
                if st.Format == 1:
                    for first, pset in zip(st.Coverage.glyphs, st.PairSet):
                        for rec in pset.PairValueRecord:
                            v = rec.Value1.XAdvance if rec.Value1 else 0
                            if v:
                                pairs.setdefault((first, rec.SecondGlyph), v)
                elif st.Format == 2:
                    c1 = st.ClassDef1.classDefs
                    c2 = st.ClassDef2.classDefs
                    cov = set(st.Coverage.glyphs)
                    for g1 in cov:
                        k1 = c1.get(g1, 0)
                        for g2, k2 in list(c2.items()) + [(None, 0)]:
                            if g2 is None:
                                continue
                            rec = st.Class1Record[k1].Class2Record[k2]
                            v = rec.Value1.XAdvance if rec.Value1 else 0
                            if v:
                                pairs.setdefault((g1, g2), v)
        return pairs

    def layout(self, text: str, size: float, tracking: float = 0.0):
        """Return [(glyphname, x_offset)] and total advance, in px."""
        s = size / self.upem
        x = 0.0
        out = []
        names = [self.cmap.get(ord(c)) for c in text]
        for i, (ch, gname) in enumerate(zip(text, names)):
            if gname is None:
                x += 0.25 * size
                continue
            out.append((gname, x))
            adv = self.glyphs[gname].width * s
            if i + 1 < len(names) and names[i + 1]:
                adv += self.kern.get((gname, names[i + 1]), 0) * s
            x += adv + tracking * size
        return out, x - tracking * size

    def glyph_path(self, gname: str, size: float, x: float, y: float) -> str:
        s = size / self.upem
        pen = SVGPathPen(self.glyphs)
        self.glyphs[gname].draw(TransformPen(pen, (s, 0, 0, -s, x, y)))
        return pen.getCommands()

    def contour_bboxes(self, gname: str, size: float, x: float, y: float):
        """Bounding boxes of each closed contour of a glyph, in px."""
        s = size / self.upem
        rec = RecordingPen()
        self.glyphs[gname].draw(TransformPen(rec, (s, 0, 0, -s, x, y)))
        boxes, cur = [], []
        for op, args in rec.value:
            if op == "moveTo" and cur:
                boxes.append(self._bbox(cur))
                cur = []
            cur.append((op, args))
        if cur:
            boxes.append(self._bbox(cur))
        return boxes

    @staticmethod
    def _bbox(ops):
        bp = BoundsPen(None)
        for op, args in ops:
            getattr(bp, op)(*args)
        if not any(op == "closePath" for op, _ in ops):
            bp.closePath()
        return bp.bounds

    def text_path(self, text, size, x, y, tracking=0.0, skip=()):
        """One <path d> for a string; returns (d, width, per-glyph boxes)."""
        placed, width = self.layout(text, size, tracking)
        ds, boxes = [], []
        for i, (g, dx) in enumerate(placed):
            boxes.append((g, x + dx, self.contour_bboxes(g, size, x + dx, y)))
            if i in skip:
                continue
            ds.append(self.glyph_path(g, size, x + dx, y))
        return " ".join(ds), width, boxes


SEMI = Face(FONT_DIR / "SourceSans3-Semibold.otf")
REG = Face(FONT_DIR / "SourceSans3-Regular.otf")


# --------------------------------------------------------------------------
# fields: kernel sums and their iso-lines
# --------------------------------------------------------------------------
def kernel_sum(centres, sigma, xs, ys, weights=None):
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i, (cx, cy) in enumerate(centres):
        w = 1.0 if weights is None else weights[i]
        Z += w * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
    return X, Y, Z


def iso_lines(X, Y, Z, levels):
    """List of (level, [polyline ndarray]) via matplotlib's contour engine."""
    fig = plt.figure()
    ax = fig.add_subplot()
    cs = ax.contour(X, Y, Z, levels=levels)
    out = []
    for lvl, path in zip(cs.levels, cs.get_paths()):
        polys = [p for p in path.to_polygons(closed_only=False) if len(p) > 2]
        out.append((lvl, polys))
    plt.close(fig)
    return out


def polyline_d(pts, close_tol=1e-6, fmt="%.1f"):
    d = "M" + " L".join(f"{fmt % x},{fmt % y}" for x, y in pts)
    if np.hypot(*(pts[0] - pts[-1])) < close_tol:
        d += "Z"
    return d


def fmt_paths(polys, project=None):
    ds = []
    for p in polys:
        q = project(p) if project else p
        closed = np.hypot(*(p[0] - p[-1])) < 1e-6
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in q)
        ds.append(d + ("Z" if closed else ""))
    return " ".join(ds)


# --------------------------------------------------------------------------
# svg helpers
# --------------------------------------------------------------------------
def svg_doc(w, h, body, defs="", bg=None, title="", rx=0):
    bg_rect = f'<rect width="{w}" height="{h}" rx="{rx}" fill="{bg}"/>' if bg else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img" aria-label="{title}">\n'
        f"<title>{title}</title>\n<defs>\n{defs}\n</defs>\n{bg_rect}\n{body}\n</svg>\n"
    )


def tagline(face, x, y, size, colour, opacity=1.0, anchor="start", tracking=0.18):
    d, w, _ = face.text_path(TAGLINE, size, 0, 0, tracking=tracking)
    if anchor == "middle":
        x -= w / 2
    return f'<path transform="translate({x:.1f},{y:.1f})" d="{d}" fill="{colour}" fill-opacity="{opacity}"/>', w


def wordmark(face, x, y, size, colour, skip=()):
    d, w, boxes = face.text_path("CORDS", size, x, y, skip=skip)
    return f'<path d="{d}" fill="{colour}"/>', w, boxes


# ==========================================================================
# Direction A - "kernel": the O carries the iso-lines of one Gaussian
# ==========================================================================
def kernel_rings(cx, cy, rx_outer, ry_outer, ink, dot, stem, dot_r, levels=(0.85, 0.62, 0.42, 0.25, 0.11)):
    """Iso-lines of a unit Gaussian, evenly spaced in *value*, so they crowd
    toward the flank exactly as real level sets do; radius r = sigma*sqrt(-2 ln l)."""
    r_max = math.sqrt(-2 * math.log(min(levels)))
    parts = []
    for l in levels:
        r = math.sqrt(-2 * math.log(l)) / r_max
        # the line fades with the field value it marks
        parts.append(
            f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx_outer * r:.1f}" '
            f'ry="{ry_outer * r:.1f}" fill="none" stroke="{ink}" '
            f'stroke-opacity="{0.42 + 0.58 * l:.2f}" stroke-width="{stem:.2f}"/>'
        )
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{dot_r:.1f}" fill="{dot}"/>')
    return "\n".join(parts)


def banner_a(dark=False):
    W, H = 1600, 500
    ink, paper = (PAPER, PAPER_DARK) if dark else (INK, PAPER)
    size = 236
    x0 = 118
    base = 302
    mark, wm_w, boxes = wordmark(SEMI, x0, base, size, ink, skip=())
    # the O: its counter is the smaller of the glyph's two contours
    g, gx, bx = boxes[1]
    inner = min(bx, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    rx = (inner[2] - inner[0]) / 2 - 11
    ry = (inner[3] - inner[1]) / 2 - 11
    rings = kernel_rings(cx, cy, rx, ry, ink, COPPER, stem=2.6, dot_r=10, levels=(0.8, 0.52, 0.28, 0.11))

    # background field: a sparse set of objects, iso-lines at low opacity,
    # kept away from the wordmark so the type stays clean.
    centres = [
        (1010, 92), (1130, 170), (1265, 118), (1390, 205), (1480, 95),
        (1190, 330), (1330, 405), (1470, 330), (1540, 445), (1080, 450),
        (300, 30), (660, 500),
    ]
    sigma = 62
    xs = np.linspace(0, W, 321)
    ys = np.linspace(0, H, 101)
    X, Y, Z = kernel_sum(centres, sigma, xs, ys)
    lines = iso_lines(X, Y, Z, levels=np.linspace(0.08, Z.max(), 12))
    field = "".join(
        f'<path d="{fmt_paths(polys)}" fill="none" stroke="{ink}" stroke-opacity="{0.16 if dark else 0.11}" stroke-width="1"/>'
        for lvl, polys in lines
    )
    dots = "".join(
        f'<circle cx="{x}" cy="{y}" r="3.2" fill="{COPPER}"/>' for x, y in centres
    )

    tag, tw = tagline(REG, x0 + 4, base + 64, 25, ink, opacity=0.72)
    body = f"""
<g>{field}</g>
<g>{dots}</g>
{mark}
<g>{rings}</g>
{tag}
"""
    return svg_doc(W, H, body, bg=paper, title="CORDS")


def mark_a(dark=True):
    S = 512
    cx = cy = 256
    R = 152
    ink, paper = (PAPER, INK) if dark else (INK, PAPER)
    rings = kernel_rings(cx, cy, R, R, ink, COPPER, stem=9, dot_r=24)
    body = f'<rect width="{S}" height="{S}" rx="112" fill="{paper}"/>\n{rings}'
    return svg_doc(S, S, body, title="CORDS mark")


# ==========================================================================
# Direction B - "planes": objects above, density and feature fields below
# ==========================================================================
def oblique(ox, oy, ax=(1.0, 0.30), ay=(0.62, -0.36)):
    """Plane coordinates (u, v) -> screen (x, y)."""
    def project(p):
        p = np.asarray(p, dtype=float)
        u, v = p[..., 0], p[..., 1]
        return np.stack([ox + ax[0] * u + ay[0] * v, oy + ax[1] * u + ay[1] * v], axis=-1)
    return project


def plane_outline(project, U, V):
    corners = np.array([(0, 0), (U, 0), (U, V), (0, V)], dtype=float)
    q = project(corners)
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in q) + "Z"


def plane_grid(project, U, V, n, stroke, opacity, width=1):
    d = []
    for i in range(1, n):
        u = U * i / n
        p = project(np.array([(u, 0), (u, V)], dtype=float))
        d.append("M%.1f,%.1f L%.1f,%.1f" % (*p[0], *p[1]))
        v = V * i / n
        p = project(np.array([(0, v), (U, v)], dtype=float))
        d.append("M%.1f,%.1f L%.1f,%.1f" % (*p[0], *p[1]))
    return f'<path d="{" ".join(d)}" fill="none" stroke="{stroke}" stroke-opacity="{opacity}" stroke-width="{width}"/>'


def planes_scene(ox, oy, U, V, dz, objects, sigma, scale=1.0, glow_r=None,
                 contour_levels=10, ink_line="#e9e4d8", line_w=1.0):
    """objects: list of (u, v, class_idx). Returns svg body."""
    proj = oblique(ox, oy)
    proj_up = oblique(ox, oy - dz)
    ax_, ay_ = (1.0, 0.30), (0.62, -0.36)
    matrix = f"matrix({ax_[0]},{ax_[1]},{ay_[0]},{ay_[1]},{ox},{oy})"
    glow_r = glow_r or 2.6 * sigma

    xs = np.linspace(0, U, 261)
    ys = np.linspace(0, V, int(261 * V / U))
    X, Y, Z = kernel_sum([(u, v) for u, v, _ in objects], sigma, xs, ys)
    lines = iso_lines(X, Y, Z, levels=np.linspace(0.1, Z.max() * 0.98, contour_levels))

    defs = []
    glows = []
    for i, (u, v, k) in enumerate(objects):
        gid = f"glow{ox}_{i}"
        defs.append(
            f'<radialGradient id="{gid}"><stop offset="0" stop-color="{CLASS[k]}" stop-opacity="0.55"/>'
            f'<stop offset="0.45" stop-color="{CLASS[k]}" stop-opacity="0.16"/>'
            f'<stop offset="1" stop-color="{CLASS[k]}" stop-opacity="0"/></radialGradient>'
        )
        glows.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="{glow_r:.1f}" fill="url(#{gid})"/>')

    clip_id = f"plane{ox}_{oy}"
    defs.append(f'<clipPath id="{clip_id}"><path d="{plane_outline(proj, U, V)}"/></clipPath>')
    lower = f'<path d="{plane_outline(proj, U, V)}" fill="#1a2028" fill-opacity="0.9" stroke="{ink_line}" stroke-opacity="0.35" stroke-width="{line_w}"/>'
    lower_grid = plane_grid(proj, U, V, 6, ink_line, 0.07, line_w)
    contours = "".join(
        f'<path d="{fmt_paths(polys, proj)}" fill="none" stroke="{ink_line}" '
        f'stroke-opacity="{0.22 + 0.5 * j / max(1, len(lines) - 1):.2f}" stroke-width="{line_w}"/>'
        for j, (lvl, polys) in enumerate(lines)
    )
    upper = f'<path d="{plane_outline(proj_up, U, V)}" fill="{ink_line}" fill-opacity="0.035" stroke="{ink_line}" stroke-opacity="0.45" stroke-width="{line_w}"/>'
    upper_grid = plane_grid(proj_up, U, V, 6, ink_line, 0.10, line_w)

    drops, objs, feet = [], [], []
    r_obj = 9 * scale
    for u, v, k in objects:
        (x1, y1), = proj_up(np.array([(u, v)], dtype=float))
        (x0, y0), = proj(np.array([(u, v)], dtype=float))
        drops.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x0:.1f}" y2="{y0:.1f}" stroke="{CLASS[k]}" stroke-opacity="0.75" stroke-width="{1.6 * scale:.1f}"/>'
        )
        feet.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="{3.2 * scale:.1f}" fill="{CLASS[k]}"/>')
        objs.append(
            f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="{r_obj:.1f}" fill="{CLASS[k]}"/>'
            f'<circle cx="{x1 - 0.3 * r_obj:.1f}" cy="{y1 - 0.3 * r_obj:.1f}" r="{0.3 * r_obj:.1f}" fill="#ffffff" fill-opacity="0.35"/>'
        )

    body = f"""
{lower}
{lower_grid}
<g clip-path="url(#{clip_id})"><g transform="{matrix}">{"".join(glows)}</g></g>
<g>{contours}</g>
<g>{"".join(feet)}</g>
<g>{"".join(drops)}</g>
{upper}
{upper_grid}
<g>{"".join(objs)}</g>
"""
    return body, "\n".join(defs)


OBJECTS_B = [
    (70, 60, 0), (150, 120, 3), (230, 70, 4), (330, 150, 1),
    (410, 60, 5), (120, 250, 2), (260, 260, 6), (380, 290, 3), (460, 220, 0),
]


def banner_b(readme=False):
    W, H = 1600, 500
    scene, defs = planes_scene(ox=90, oy=330, U=480, V=320, dz=140,
                               objects=OBJECTS_B, sigma=46)
    type_col = "#f2eee6"
    if readme:
        # two tracked lines under the wordmark stay legible at README width
        size = 228
        base = 268
        _, wm_w = SEMI.layout("CORDS", size)
        x0 = W - 104 - wm_w
        mark, wm_w, _ = wordmark(SEMI, x0, base, size, type_col)
        tsize = 30
        lines = ["CONTINUOUS REPRESENTATIONS", "OF DISCRETE STRUCTURES"]
        tag = ""
        for i, line in enumerate(lines):
            d, w, _ = REG.text_path(line, tsize, 0, 0, tracking=0.16)
            tag += (f'<path transform="translate({x0 + 5:.1f},{base + 58 + i * 44:.1f})" '
                    f'd="{d}" fill="{type_col}" fill-opacity="0.62"/>')
        body = f"""
{scene}
{mark}
{tag}
"""
        return svg_doc(W, H, body, defs=defs, bg=PAPER_DARK, title="CORDS", rx=28)
    base = 292
    size = 206
    tsize = 19.5
    _, tw = REG.layout(TAGLINE, tsize, tracking=0.18)
    x0 = W - 100 - tw
    mark, wm_w, _ = wordmark(SEMI, x0, base, size, type_col)
    tag, tw = tagline(REG, x0 + 4, base + 56, tsize, type_col, opacity=0.6)
    body = f"""
{scene}
{mark}
{tag}
"""
    return svg_doc(W, H, body, defs=defs, bg=PAPER_DARK, title="CORDS")


def mark_b():
    S = 512
    objects = [(70, 60, 0), (190, 140, 3), (110, 210, 5)]
    scene, defs = planes_scene(ox=82, oy=352, U=260, V=210, dz=150,
                               objects=objects, sigma=40, scale=1.6,
                               contour_levels=7, line_w=1.6)
    body = f'<rect width="{S}" height="{S}" rx="112" fill="{PAPER_DARK}"/>\n{scene}'
    return svg_doc(S, S, body, defs=defs, title="CORDS mark")


# ==========================================================================
# Direction C - "comb": stems of unit mass and the smooth sum that carries them
# ==========================================================================
def comb_scene(x_left, x_right, base, positions, sigma, stem_h,
               accent, ink, stem_w=2.4, dot_r=6.5, curve_w=3.2, gid="c",
               stem_fade=True, curve_fade=True, area_opacity=0.14):
    xs = np.linspace(x_left, x_right, 700)
    rho = np.zeros_like(xs)
    for p in positions:
        rho += np.exp(-((xs - p) ** 2) / (2 * sigma**2))
    # one unit of mass is one stem height: an isolated object's kernel peaks
    # exactly at its stem, overlapping objects sum above their stems.
    y = base - rho * stem_h
    curve = "M" + " L".join(f"{x:.1f},{v:.1f}" for x, v in zip(xs, y))
    area = curve + f" L{x_right:.1f},{base:.1f} L{x_left:.1f},{base:.1f}Z"

    stems = "".join(
        f'<line x1="{p:.1f}" y1="{base:.1f}" x2="{p:.1f}" y2="{base - stem_h:.1f}" stroke="{accent}" stroke-width="{stem_w}"/>'
        f'<circle cx="{p:.1f}" cy="{base - stem_h:.1f}" r="{dot_r}" fill="{accent}"/>'
        for p in positions
    )
    defs = f"""
<linearGradient id="{gid}_area" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="{ink}" stop-opacity="{area_opacity}"/>
  <stop offset="1" stop-color="{ink}" stop-opacity="0.0"/>
</linearGradient>
<linearGradient id="{gid}_lr" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#fff" stop-opacity="0.18"/>
  <stop offset="0.35" stop-color="#fff" stop-opacity="0.35"/>
  <stop offset="1" stop-color="#fff" stop-opacity="1"/>
</linearGradient>
<linearGradient id="{gid}_rl" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#fff" stop-opacity="1"/>
  <stop offset="0.65" stop-color="#fff" stop-opacity="0.5"/>
  <stop offset="1" stop-color="#fff" stop-opacity="0.22"/>
</linearGradient>
<mask id="{gid}_mlr"><rect x="{x_left - 20}" y="0" width="{x_right - x_left + 40}" height="2000" fill="url(#{gid}_lr)"/></mask>
<mask id="{gid}_mrl"><rect x="{x_left - 20}" y="0" width="{x_right - x_left + 40}" height="2000" fill="url(#{gid}_rl)"/></mask>
"""
    stems_group = f'<g mask="url(#{gid}_mrl)">{stems}</g>' if stem_fade else f"<g>{stems}</g>"
    curve_mask = f' mask="url(#{gid}_mlr)"' if curve_fade else ""
    body = f"""
<g{curve_mask}>
  <path d="{area}" fill="url(#{gid}_area)"/>
  <path d="{curve}" fill="none" stroke="{ink}" stroke-width="{curve_w}" stroke-linejoin="round"/>
</g>
{stems_group}
<line x1="{x_left:.1f}" y1="{base:.1f}" x2="{x_right:.1f}" y2="{base:.1f}" stroke="{ink}" stroke-opacity="0.35" stroke-width="1.2"/>
"""
    return body, defs


POSITIONS_C = [905, 948, 1040, 1128, 1160, 1192, 1298, 1345, 1436, 1466, 1498]


def banner_c(dark=False):
    W, H = 1600, 500
    ink, paper = (PAPER, PAPER_DARK) if dark else (INK, PAPER)
    x0 = 118
    base = 296
    size = 236
    mark, wm_w, _ = wordmark(SEMI, x0, base, size, ink)
    tag, tw = tagline(REG, x0 + 4, base + 64, 25, ink, opacity=0.72)
    scene, defs = comb_scene(
        x_left=870, x_right=1530, base=base, positions=POSITIONS_C, sigma=36,
        stem_h=96, accent=TEAL, ink=ink, area_opacity=0.16 if dark else 0.13,
    )
    body = f"""
{scene}
{mark}
{tag}
"""
    return svg_doc(W, H, body, defs=defs, bg=paper, title="CORDS")


def mark_c(dark=True):
    S = 512
    ink, paper = (PAPER, INK) if dark else (INK, PAPER)
    base = 346
    positions = [150, 186, 268, 338, 366]
    scene, defs = comb_scene(
        x_left=88, x_right=424, base=base, positions=positions, sigma=27,
        stem_h=92, accent=TEAL, ink=ink, stem_w=6, dot_r=12,
        curve_w=8, gid="m", stem_fade=False, curve_fade=False, area_opacity=0.10,
    )
    body = f'<rect width="{S}" height="{S}" rx="112" fill="{paper}"/>\n{scene}'
    return svg_doc(S, S, body, defs=defs, title="CORDS mark")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    files = {
        "cords-banner-A-kernel.svg": banner_a(),
        "cords-banner-A-kernel-dark.svg": banner_a(dark=True),
        "cords-mark-A-kernel.svg": mark_a(),
        "cords-mark-A-kernel-light.svg": mark_a(dark=False),
        "cords-banner-B-planes.svg": banner_b(),
        "cords-banner.svg": banner_b(readme=True),
        "cords-mark-B-planes.svg": mark_b(),
        "cords-banner-C-comb.svg": banner_c(),
        "cords-banner-C-comb-dark.svg": banner_c(dark=True),
        "cords-mark-C-comb.svg": mark_c(),
        "cords-mark-C-comb-light.svg": mark_c(dark=False),
    }
    for name, svg in files.items():
        (OUT / name).write_text(svg)
        print(f"{name:32s} {len(svg) / 1024:6.1f} kB")
    print("accents:", COPPER, TEAL, CLASS)
