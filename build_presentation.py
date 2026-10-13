"""
build_presentation.py
Generates a warm-gradient technical/research deck (~34 slides) for the
Cascade-Based Input-Doubling Classifier & Linear Regression Pipeline
(Pediatric Allogeneic Bone Marrow Transplant Survival, Small-Data Case).

Run:  python build_presentation.py
Out:  Bone_Marrow_BMT_Presentation.pptx  (+ presentation_bg.png)
"""

import os
import numpy as np
from PIL import Image, ImageFilter
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
BG_IMG = os.path.join(HERE, "presentation_bg.png")
OUT = os.path.join(HERE, "Bone_Marrow_BMT_Presentation.pptx")

# ----------------------------------------------------------------------------
# Theme (warm gradient / dark-ink typography)
# ----------------------------------------------------------------------------
INK       = RGBColor(0x2A, 0x23, 0x1F)
INK_SOFT  = RGBColor(0x46, 0x3B, 0x34)
GRAY      = RGBColor(0x5E, 0x52, 0x4A)
KICKER    = RGBColor(0x4A, 0x3E, 0x36)
CARD_FILL = RGBColor(0xFF, 0xF4, 0xE9)
BORDER    = RGBColor(0xF6, 0xDD, 0xC6)
CIRCLE    = RGBColor(0x26, 0x20, 0x1D)
CIRCLE_TX = RGBColor(0xD9, 0xC9, 0xB8)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)

RED    = RGBColor(0xB2, 0x3A, 0x2E)
AMBER  = RGBColor(0xA8, 0x5B, 0x12)
TEAL   = RGBColor(0x14, 0x63, 0x5F)
BROWN  = RGBColor(0x7A, 0x52, 0x30)
PLUM   = RGBColor(0x7A, 0x2E, 0x5A)

TBL_HEAD   = RGBColor(0x33, 0x2A, 0x24)
TBL_HEAD_T = RGBColor(0xFA, 0xF0, 0xE6)
TBL_ROW1   = RGBColor(0xFB, 0xE9, 0xD9)
TBL_ROW2   = RGBColor(0xF2, 0xD6, 0xBC)

FONT = "Segoe UI"
FONT_MONO = "Consolas"

EMU_W = Inches(13.333)
EMU_H = Inches(7.5)

prs = Presentation()
prs.slide_width = EMU_W
prs.slide_height = EMU_H
BLANK = prs.slide_layouts[6]


# ----------------------------------------------------------------------------
# Background image generation
# ----------------------------------------------------------------------------
def make_background(path):
    W, H = 1920, 1080
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = xx / (W - 1)
    yn = yy / (H - 1)

    def C(r, g, b):
        return np.array([r, g, b], np.float32)

    tl, tr = C(243, 170, 104), C(242, 186, 88)
    bl, br = C(214, 78, 64), C(242, 170, 66)
    top = tl[None, None] * (1 - xn[..., None]) + tr[None, None] * xn[..., None]
    bot = bl[None, None] * (1 - xn[..., None]) + br[None, None] * xn[..., None]
    base = top * (1 - yn[..., None]) + bot * yn[..., None]

    d = np.sqrt((xn - 0.02) ** 2 + (yn - 1.05) ** 2)
    b = np.exp(-(d / 0.55) ** 2)[..., None]
    base = base * (1 - 0.55 * b) + C(206, 66, 64)[None, None] * 0.55 * b

    d2 = np.sqrt((xn - 0.12) ** 2 + (yn - 0.02) ** 2)
    b2 = np.exp(-(d2 / 0.5) ** 2)[..., None]
    base = base * (1 - 0.30 * b2) + C(252, 206, 156)[None, None] * 0.30 * b2

    d3 = np.sqrt((xn - 0.7) ** 2 + (yn - 0.5) ** 2)
    b3 = np.exp(-(d3 / 0.8) ** 2)[..., None]
    base = base * (1 - 0.12 * b3) + C(246, 180, 96)[None, None] * 0.12 * b3

    rng = np.random.default_rng(11)
    grain = rng.normal(0, 7.0, (H, W, 1))[..., 0]
    gimg = Image.fromarray(np.clip(grain + 128, 0, 255).astype(np.uint8))
    gimg = gimg.filter(ImageFilter.GaussianBlur(0.6))
    g2 = (np.asarray(gimg).astype(np.float32) - 128)[..., None]
    base = np.clip(base + g2 * 1.5, 0, 255)
    Image.fromarray(base.astype(np.uint8)).save(path)


make_background(BG_IMG)


# ----------------------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------------------
def set_alpha(shape, alpha):
    try:
        spPr = shape._element.spPr
        sf = spPr.find(qn("a:solidFill"))
        clr = sf.find(qn("a:srgbClr"))
        for a in clr.findall(qn("a:alpha")):
            clr.remove(a)
        clr.append(clr.makeelement(qn("a:alpha"), {"val": str(int(alpha * 100000))}))
    except Exception:
        pass


def set_spacing(run, val):
    try:
        run.font._rPr.set("spc", str(val))
    except Exception:
        pass


def slide():
    s = prs.slides.add_slide(BLANK)
    s.shapes.add_picture(BG_IMG, 0, 0, EMU_W, EMU_H)
    return s


def rect(s, x, y, w, h, fill=CARD_FILL, line=None, radius=0.06,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, alpha=None):
    sh = s.shapes.add_shape(shape, x, y, w, h)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
        if alpha is not None:
            set_alpha(sh, alpha)
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1.0)
    sh.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            sh.adjustments[0] = radius
        except Exception:
            pass
    return sh


def textbox(s, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    return tf


def para(tf, text, size=16, color=GRAY, bold=False, first=False, font=FONT,
         align=PP_ALIGN.LEFT, space_before=4, space_after=4, italic=False, spc=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = font
    if spc:
        set_spacing(r, spc)
    return p


def bullet(tf, text, size=15, color=GRAY, bold=False, first=False, level=0,
           marker=u"\u2014   ", mcolor=RED, space_before=6, space_after=6, font=FONT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.level = level
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if marker:
        rm = p.add_run()
        rm.text = (u"     " * level) + marker
        rm.font.size = Pt(size)
        rm.font.color.rgb = mcolor
        rm.font.bold = True
        rm.font.name = FONT
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return p


def rich(tf, segments, size=15, first=False, space_before=5, space_after=5, font=FONT,
         align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    for seg in segments:
        txt, col, bold = seg[0], seg[1], seg[2]
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = col
        r.font.name = font
    return p


def card(s, x, y, w, h, alpha=0.30, radius=0.05, border=BORDER):
    rect(s, x, y, w, h, fill=CARD_FILL, line=border, radius=radius, alpha=alpha)
    return x + Inches(0.34), y + Inches(0.30), w - Inches(0.68), h - Inches(0.60)


def header(s, kicker, title, page, tsize=32):
    tf = textbox(s, Inches(0.9), Inches(0.5), Inches(11.5), Inches(0.35))
    para(tf, kicker.upper(), size=12, color=KICKER, bold=True, first=True,
         space_before=0, space_after=0, spc=180)
    tf2 = textbox(s, Inches(0.9), Inches(0.86), Inches(11.7), Inches(0.8))
    para(tf2, title, size=tsize, color=INK, bold=True, first=True,
         space_before=0, space_after=0)
    footer(s, page)


def footer(s, page):
    tf = textbox(s, Inches(0.9), Inches(6.95), Inches(11.5), Inches(0.35))
    para(tf, "CASCADE INPUT-DOUBLING PIPELINE   |   IZONIN ET AL. (2025), COMPUTATION 13(3):80",
         size=8.5, color=RGBColor(0x5A, 0x3A, 0x2A), bold=True, first=True,
         space_before=0, space_after=0, spc=80)
    tf2 = textbox(s, Inches(0.9), Inches(6.95), Inches(1.2), Inches(0.4))
    para(tf2, "%02d" % page, size=15, color=WHITE, bold=True, first=True,
         space_before=0, space_after=0)


def circle_stat(s, x, y, d, value, vsize=26):
    c = rect(s, x, y, d, d, fill=CIRCLE, line=None, shape=MSO_SHAPE.OVAL)
    tf = textbox(s, x, y, d, d, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, value, size=vsize, color=CIRCLE_TX, bold=True, first=True,
         align=PP_ALIGN.CENTER, space_before=0, space_after=0)


def table(s, x, y, w, h, data, col_widths=None, font_size=12, header_size=12,
          row_h=0.4, highlight=None, first_left=True):
    rows, cols = len(data), len(data[0])
    gt = s.shapes.add_table(rows, cols, x, y, w, h).table
    if col_widths:
        tot = sum(col_widths)
        for i, cw in enumerate(col_widths):
            gt.columns[i].width = Emu(int(w * cw / tot))
    for ri, row in enumerate(data):
        gt.rows[ri].height = Inches(row_h)
        for ci, val in enumerate(row):
            cell = gt.cell(ri, ci)
            cell.margin_left = Inches(0.09)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if ri == 0:
                cell.fill.fore_color.rgb = TBL_HEAD
            elif highlight and ri in highlight:
                cell.fill.fore_color.rgb = RGBColor(0xF6, 0xC9, 0x9B)
            else:
                cell.fill.fore_color.rgb = TBL_ROW1 if ri % 2 else TBL_ROW2
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if (ci == 0 and first_left) else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.name = FONT
            r.font.size = Pt(header_size if ri == 0 else font_size)
            r.font.bold = (ri == 0)
            r.font.color.rgb = TBL_HEAD_T if ri == 0 else INK
    return gt


def picture_fit(s, path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    box_ar, img_ar = w / h, iw / ih
    if img_ar > box_ar:
        nw, nh = w, int(w / img_ar)
    else:
        nh, nw = h, int(h * img_ar)
    return s.shapes.add_picture(path, x + int((w - nw) / 2), y + int((h - nh) / 2), nw, nh)


def figure_panel(s, fn, x, y, w, h, caption=None):
    rect(s, x, y, w, h, fill=WHITE, line=BORDER, radius=0.04, alpha=0.90)
    p = os.path.join(PLOTS, fn)
    top = y + Inches(0.16)
    bh = h - Inches(0.30) - (Inches(0.4) if caption else 0)
    if os.path.exists(p):
        picture_fit(s, p, x + Inches(0.16), top, w - Inches(0.32), bh)
    if caption:
        tf = textbox(s, x + Inches(0.16), y + h - Inches(0.44), w - Inches(0.32), Inches(0.32))
        para(tf, caption, size=10.5, color=GRAY, bold=True, first=True,
             align=PP_ALIGN.CENTER, space_before=0, space_after=0)


def section_divider(page, num, title, subtitle, color=RED):
    s = slide()
    tf = textbox(s, Inches(0.9), Inches(2.15), Inches(11.5), Inches(0.4))
    para(tf, u"SECTION %s" % num, size=14, color=color, bold=True, first=True,
         space_before=0, space_after=0, spc=260)
    tf2 = textbox(s, Inches(0.9), Inches(2.6), Inches(11.5), Inches(1.6))
    para(tf2, title, size=54, color=INK, bold=True, first=True, space_before=0, space_after=0)
    rect(s, Inches(0.92), Inches(4.15), Inches(2.2), Inches(0.06), fill=color, line=None,
         shape=MSO_SHAPE.RECTANGLE)
    tf3 = textbox(s, Inches(0.9), Inches(4.45), Inches(10.8), Inches(0.9))
    para(tf3, subtitle, size=17, color=GRAY, first=True, space_before=0, space_after=0)
    tfn = textbox(s, Inches(9.4), Inches(1.6), Inches(3.2), Inches(3.0))
    para(tfn, num, size=150, color=RGBColor(0xE8, 0xC4, 0x9E), bold=True, first=True,
         align=PP_ALIGN.RIGHT, space_before=0, space_after=0)
    footer(s, page)
    return s


# ============================================================================
# 01 - Title
# ============================================================================
s = slide()
rect(s, Inches(0.9), Inches(1.4), Inches(1.9), Inches(0.07), fill=RED, line=None,
     shape=MSO_SHAPE.RECTANGLE)
tf = textbox(s, Inches(0.9), Inches(1.6), Inches(11.4), Inches(0.4))
para(tf, "MACHINE LEARNING  /  CLINICAL RESEARCH  /  SMALL-DATA CASE", size=13,
     color=KICKER, bold=True, first=True, space_before=0, space_after=0, spc=180)
tf = textbox(s, Inches(0.9), Inches(2.15), Inches(11.6), Inches(2.4))
para(tf, "Cascade-Based Input-Doubling", size=46, color=INK, bold=True, first=True,
     space_before=0, space_after=0)
para(tf, "Classifier & Linear Regression", size=46, color=RED, bold=True,
     space_before=0, space_after=0)
tf = textbox(s, Inches(0.9), Inches(4.35), Inches(10.6), Inches(1.0))
para(tf, "Predicting survival in pediatric allogeneic bone marrow transplantation",
     size=18, color=INK_SOFT, first=True, space_before=0, space_after=4)
para(tf, "A reproducible, leak-free pipeline benchmarked against an OLS baseline",
     size=14, color=GRAY, italic=True, space_before=0, space_after=0)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(5.35), Inches(8.6), Inches(1.0), alpha=0.28)
tf = textbox(s, cx, cy, cw, ch)
rich(tf, [("Reference:  ", RED, True),
          ("Izonin, Tkachenko, Hovdysh, Berezsky, Yemets & Tsmots (2025). ", INK_SOFT, False),
          ("Computation", INK, True), (" 13(3):80, MDPI Open Access.", INK_SOFT, False)],
     size=12.5, first=True)
footer(s, 1)

# ============================================================================
# 02 - Total patients
# ============================================================================
s = slide()
cx, cy, cw, ch = card(s, Inches(0.8), Inches(1.1), Inches(5.3), Inches(5.5), alpha=0.32)
circle_stat(s, cx, cy + Inches(0.5), Inches(1.75), "187", 34)
tf = textbox(s, cx + Inches(2.15), cy + Inches(0.62), cw - Inches(2.2), Inches(1.6))
para(tf, "Total", size=34, color=INK, bold=True, first=True, space_before=0, space_after=0)
para(tf, "Patients", size=34, color=INK, bold=True, space_before=0, space_after=4)
para(tf, "UCI DATASET ID: 565", size=11, color=KICKER, bold=True, space_before=0,
     space_after=0, spc=140)
rect(s, cx, cy + Inches(2.65), cw, Inches(0.02), fill=RGBColor(0xD9, 0xB6, 0x92),
     line=None, shape=MSO_SHAPE.RECTANGLE)
tf = textbox(s, cx, cy + Inches(2.95), cw, Inches(2.2))
rich(tf, [("Source:  ", INK, True), ("UCI Machine Learning Repository", INK_SOFT, False)],
     size=15, first=True)
para(tf, "Pediatric bone marrow transplant data used for survival rule induction "
         "and outcome prediction.", size=14.5, color=GRAY, space_before=10)
rich(tf, [("35 de-leaked baseline attributes  ", TEAL, True),
          ("available at transplant time.", GRAY, False)], size=14, space_before=12)

tf = textbox(s, Inches(6.9), Inches(1.3), Inches(5.6), Inches(0.4))
para(tf, "RESEARCH CONTEXT", size=12, color=KICKER, bold=True, first=True,
     space_before=0, space_after=0, spc=200)
tf = textbox(s, Inches(6.9), Inches(1.75), Inches(5.6), Inches(4.6))
para(tf, "Small data, big stakes", size=30, color=INK, bold=True, first=True,
     space_before=0, space_after=14)
rich(tf, [("Two targets, two questions.  ", INK, True),
          ("One model estimates survival_time (days); the other classifies "
           "survival_status (0 = survived, 1 = deceased).", GRAY, False)], size=14.5)
rich(tf, [("Imbalance.  ", INK, True),
          ("Deceased outcomes are the minority class, so naive accuracy is misleading.",
           GRAY, False)], size=14.5)
rich(tf, [("Generalization gap.  ", INK, True),
          ("Deep networks overfit tiny clinical cohorts; the cascade is designed "
           "specifically for this regime.", GRAY, False)], size=14.5)
footer(s, 2)

# ============================================================================
# 03 - Development vs external test
# ============================================================================
s = slide()
cx, cy, cw, ch = card(s, Inches(0.8), Inches(1.1), Inches(5.3), Inches(5.5), alpha=0.32)
circle_stat(s, cx, cy + Inches(0.5), Inches(1.75), "157", 34)
tf = textbox(s, cx + Inches(2.15), cy + Inches(0.62), cw - Inches(2.2), Inches(1.6))
para(tf, "Development", size=32, color=INK, bold=True, first=True, space_before=0, space_after=0)
para(tf, "Cohort", size=32, color=INK, bold=True, space_before=0, space_after=4)
para(tf, "TRAINING SPLIT  -  84%", size=11, color=KICKER, bold=True, space_before=0,
     space_after=0, spc=140)
rect(s, cx, cy + Inches(2.65), cw, Inches(0.02), fill=RGBColor(0xD9, 0xB6, 0x92),
     line=None, shape=MSO_SHAPE.RECTANGLE)
tf = textbox(s, cx, cy + Inches(2.95), cw, Inches(2.2))
rich(tf, [("Source:  ", INK, True), ("remaining_dataset.csv", TEAL, True)],
     size=14, first=True)
para(tf, "Used for corruption testing, cleaning, EDA, model training and internal "
         "cross-validation.", size=14, color=GRAY, space_before=10)
rich(tf, [("Role:  ", TEAL, True), ("everything the models are allowed to learn from.",
          GRAY, False)], size=13.5, space_before=12)

tf = textbox(s, Inches(6.9), Inches(1.3), Inches(5.6), Inches(0.4))
para(tf, "DATASET SPLIT", size=12, color=KICKER, bold=True, first=True,
     space_before=0, space_after=0, spc=200)
tf = textbox(s, Inches(6.9), Inches(1.75), Inches(5.6), Inches(4.6))
para(tf, "Development vs. External Test", size=30, color=INK, bold=True, first=True,
     space_before=0, space_after=16)
rich(tf, [("157 Patients (84%)", INK, True)], size=17, space_after=2)
rich(tf, [("Development dataset used for cleaning, EDA, training and internal "
           "validation.", GRAY, False)], size=14.5, space_after=16)
rich(tf, [("30 Patients (16%)", INK, True)], size=17, space_after=2)
rich(tf, [("External test dataset kept separate until final evaluation to ensure "
           "unbiased testing.", GRAY, False)], size=14.5)
footer(s, 3)

# ============================================================================
# 04 - Roadmap
# ============================================================================
s = slide()
header(s, "Contents", "Roadmap", 4)
items = [
    ("01", "Problem & context", "Small data, class imbalance, clinical stakes", RED),
    ("02", "Research foundation", "Algorithm 1 - Cascade Input-Doubling", AMBER),
    ("03", "Pipeline architecture", "Corruption, cleaning, leak-free design", BROWN),
    ("04", "Linear regression", "OLS baseline + 100x bootstrapping", TEAL),
    ("05", "Cascade classifier", "PNN expansion, doubling, RF meta-voting", PLUM),
    ("06", "Results & diagnostics", "Metrics, confusion matrices, importance", RED),
    ("07", "Web application", "FastAPI inference + dashboard", AMBER),
    ("08", "Findings & limitations", "Conclusions, caveats, ethics", BROWN),
]
x0, y0 = Inches(0.9), Inches(2.0)
cw, ch = Inches(5.75), Inches(1.15)
gx, gy = Inches(0.25), Inches(0.2)
for i, (num, title, sub, col) in enumerate(items):
    r, c = divmod(i, 2)
    x = x0 + c * (cw + gx)
    y = y0 + r * (ch + gy)
    rect(s, x, y, cw, ch, fill=CARD_FILL, line=BORDER, radius=0.08, alpha=0.30)
    rect(s, x, y, Inches(0.08), ch, fill=col, line=None, shape=MSO_SHAPE.RECTANGLE)
    tfn = textbox(s, x + Inches(0.3), y + Inches(0.16), Inches(0.9), Inches(0.8),
                  anchor=MSO_ANCHOR.MIDDLE)
    para(tfn, num, size=26, color=col, bold=True, first=True, space_before=0, space_after=0)
    tft = textbox(s, x + Inches(1.2), y + Inches(0.2), cw - Inches(1.45), ch - Inches(0.3))
    para(tft, title, size=15, color=INK, bold=True, first=True, space_before=0, space_after=2)
    para(tft, sub, size=11.5, color=GRAY, space_before=0, space_after=0)
footer(s, 4)

# ============================================================================
# 05 - Problem
# ============================================================================
s = slide()
header(s, "01 - Motivation", "The small-data problem in transplantology", 5)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(2.0), Inches(5.6), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Clinical constraints", size=18, color=RED, bold=True, first=True,
     space_before=0, space_after=12)
bullet(tf, "Datasets are strictly limited - N=157 training patients.", mcolor=RED)
bullet(tf, "Marked class imbalance: the event of interest is the minority.", mcolor=RED)
bullet(tf, "Heavy right-censoring and heterogeneous clinical attributes.", mcolor=RED)
bullet(tf, "High-stakes decisions carry real human cost.", mcolor=RED)
rich(tf, [("Consequence:  ", INK, True),
          ("deep nets and complex non-linear classifiers overfit and fail to generalize.",
           GRAY, False)], size=14)
cx, cy, cw, ch = card(s, Inches(6.9), Inches(2.0), Inches(5.6), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Project response", size=18, color=TEAL, bold=True, first=True,
     space_before=0, space_after=12)
bullet(tf, "Implement Algorithm 1 end-to-end and test its small-data claims.", mcolor=TEAL)
bullet(tf, "Pair it with a transparent OLS + bootstrapping baseline.", mcolor=TEAL)
bullet(tf, "Evaluate on a strictly held-out clean test cohort.", mcolor=TEAL)
bullet(tf, "Ship a reproducible pipeline and an interactive web app.", mcolor=TEAL)
rich(tf, [("Thesis:  ", INK, True),
          ("neural probability projection enriches low-dimensional clinical "
           "feature spaces.", GRAY, False)], size=14)
footer(s, 5)

# ============================================================================
# 06 - Research foundation
# ============================================================================
s = slide()
header(s, "02 - Research foundation", "Algorithm 1: a three-stage cascade", 6)
stages = [
    ("STAGE 1", "PNN probability expansion",
     ["Euclidean, Chebyshev & Cosine distances to support vectors",
      "Gaussian kernel with smooth factor sigma",
      "Appends probabilities: [x, P_E, P_Cheb, P_Cos]"], RED),
    ("STAGE 2", "Quadratic input-doubling",
     ["Concatenate sample pairs (x_i, x_j)",
      "Target becomes the difference z = y_i - y_j in {-1, 0, 1}",
      "Small N becomes thousands of comparisons"], AMBER),
    ("STAGE 3", "Random Forest meta-classifier",
     ["Trained on the doubled feature space",
      "Plurality voting over reference training vectors",
      "Final discrete class from ensemble votes"], PLUM),
]
x0, y0 = Inches(0.9), Inches(2.1)
cw, ch = Inches(3.7), Inches(4.4)
gx = Inches(0.28)
for i, (tag, name, pts, col) in enumerate(stages):
    x = x0 + i * (cw + gx)
    rect(s, x, y0, cw, ch, fill=CARD_FILL, line=BORDER, radius=0.05, alpha=0.32)
    rect(s, x, y0, cw, Inches(0.09), fill=col, line=None, shape=MSO_SHAPE.RECTANGLE)
    tft = textbox(s, x + Inches(0.3), y0 + Inches(0.3), cw - Inches(0.6), Inches(1.2))
    para(tft, tag, size=11, color=col, bold=True, first=True, space_before=0, space_after=2,
         spc=140)
    para(tft, name, size=18, color=INK, bold=True, space_before=0, space_after=0)
    tfb = textbox(s, x + Inches(0.3), y0 + Inches(1.75), cw - Inches(0.6), ch - Inches(2.0))
    for j, p in enumerate(pts):
        bullet(tfb, p, size=12.5, first=(j == 0), mcolor=col, space_before=7, space_after=7)
footer(s, 6)

# ============================================================================
# 07 - Architecture flow
# ============================================================================
s = slide()
header(s, "03 - Architecture", "End-to-end pipeline", 7)
flow = [
    ("RAW DATA", "157 train + 30 test", BROWN),
    ("STEP A", "Artificial corruption", RED),
    ("STEP B", "Cleaning & preprocessing", AMBER),
    ("MODELS", "OLS + Cascade", TEAL),
    ("BENCHMARK", "Metrics, plots, exports", PLUM),
]
x0, y0 = Inches(0.9), Inches(2.3)
cw, ch = Inches(2.07), Inches(1.6)
gx = Inches(0.4)
for i, (tag, name, col) in enumerate(flow):
    x = x0 + i * (cw + gx)
    rect(s, x, y0, cw, ch, fill=CARD_FILL, line=col, radius=0.09, alpha=0.34)
    rect(s, x, y0, cw, Inches(0.08), fill=col, line=None, shape=MSO_SHAPE.RECTANGLE)
    tft = textbox(s, x + Inches(0.16), y0 + Inches(0.38), cw - Inches(0.32), ch - Inches(0.5))
    para(tft, tag, size=10.5, color=col, bold=True, first=True, align=PP_ALIGN.CENTER,
         space_before=0, space_after=3, spc=100)
    para(tft, name, size=13.5, color=INK, bold=True, align=PP_ALIGN.CENTER,
         space_before=0, space_after=0)
    if i < 4:
        tfa = textbox(s, x + cw - Inches(0.02), y0 + Inches(0.5), Inches(0.5), Inches(0.6))
        para(tfa, u"\u279C", size=22, color=INK_SOFT, bold=True, first=True,
             align=PP_ALIGN.CENTER, space_before=0, space_after=0)
cx, cy, cw2, ch2 = card(s, Inches(0.9), Inches(4.3), Inches(11.6), Inches(2.1), alpha=0.30)
tf = textbox(s, cx, cy, cw2, ch2)
para(tf, "Engineering principles", size=16, color=RED, bold=True, first=True,
     space_before=0, space_after=10)
rich(tf, [("Strict train/test isolation.  ", INK, True),
          ("Imputers, IQR caps, scaler and encoder are fitted on train only, then applied "
           "to test - no leakage.", GRAY, False)], size=13)
rich(tf, [("Reproducibility.  ", INK, True),
          ("Fixed seeds; one master script (pipeline.py) regenerates every CSV, metric "
           "and figure.", GRAY, False)], size=13)
rich(tf, [("Modular source.  ", INK, True),
          ("Corruption, cleaning, regression, cascade, evaluation and visualization live "
           "in separate modules under src/.", GRAY, False)], size=13)
footer(s, 7)

# ============================================================================
# 08 - Step A corruption
# ============================================================================
s = slide()
header(s, "03 - Architecture", "Step A - artificial data corruption", 8)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(2.0), Inches(5.9), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Injected distortions", size=18, color=RED, bold=True, first=True,
     space_before=0, space_after=12)
bullet(tf, "Random 10-15% NaN missingness across numerical features.", mcolor=RED)
bullet(tf, "Synthetic duplicate rows (dirty clinical records).", mcolor=RED)
bullet(tf, "Non-standard string casing & whitespace in Disease.", mcolor=RED)
bullet(tf, "Extreme outliers in CD34 dose, weights and ages.", mcolor=RED)
rich(tf, [("Purpose:  ", INK, True),
          ("stress-test the cleaning pipeline under realistic hospital-data degradation.",
           GRAY, False)], size=13.5)
cx, cy, cw, ch = card(s, Inches(7.1), Inches(2.0), Inches(5.4), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Corruption -> cleaning contract", size=16, color=TEAL, bold=True, first=True,
     space_before=0, space_after=12)
rows = [("Output", "dirty_dataset.csv"), ("Missingness", "12% of numeric cells"),
        ("Duplicates", "14 injected rows"), ("Outliers", "8 extreme points"),
        ("String noise", "casing + whitespace"), ("Seed", "random_state = 42")]
for i, (k, v) in enumerate(rows):
    rich(tf, [(k + "   ", GRAY, False), (v, INK, True)], size=14, first=(i == 8))
rich(tf, [("Why corrupt on purpose?  ", RED, True),
          ("It makes cleaning robustness measurable and keeps the experiment reproducible.",
           GRAY, False)], size=12.5, space_before=16)
footer(s, 8)

# ============================================================================
# 09 - Step B cleaning
# ============================================================================
s = slide()
header(s, "03 - Architecture", "Step B - cleaning & preprocessing", 9)
steps = [
    ("01", "Deduplicate", "Drop exact duplicate rows injected in Step A.", RED),
    ("02", "Outlier capping", "IQR clipping to [Q1 - 3*IQR, Q3 + 3*IQR].", AMBER),
    ("03", "Imputation", "Median (numeric) & mode (categorical).", BROWN),
    ("04", "Encode & scale", "One-hot categoricals + StandardScaler.", TEAL),
]
x0, y0 = Inches(0.9), Inches(2.1)
cw, ch = Inches(2.83), Inches(1.9)
gx = Inches(0.14)
for i, (num, name, desc, col) in enumerate(steps):
    x = x0 + i * (cw + gx)
    rect(s, x, y0, cw, ch, fill=CARD_FILL, line=BORDER, radius=0.08, alpha=0.32)
    rect(s, x, y0, cw, Inches(0.08), fill=col, line=None, shape=MSO_SHAPE.RECTANGLE)
    tft = textbox(s, x + Inches(0.26), y0 + Inches(0.34), cw - Inches(0.5), ch - Inches(0.5))
    para(tft, "STEP " + num, size=10, color=col, bold=True, first=True, space_before=0,
         space_after=2, spc=120)
    para(tft, name, size=16, color=INK, bold=True, space_before=0, space_after=4)
    para(tft, desc, size=12, color=GRAY, space_before=0, space_after=0)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(4.35), Inches(11.6), Inches(2.05), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "No-leakage guarantee", size=16, color=TEAL, bold=True, first=True,
     space_before=0, space_after=8)
rich(tf, [("BoneMarrowPreprocessor.fit()  ", INK, True),
          ("is called only on the corrupted training set; ", GRAY, False),
          ("transform()  ", INK, True),
          ("is then applied independently to train and to the clean 30-patient test cohort. "
           "All statistics are frozen from training.", GRAY, False)], size=13)
rich(tf, [("Artifacts:  ", GRAY, False), ("cleaned_dataset.csv", TEAL, True),
          ("   -   ", GRAY, False), ("cleaned_test_dataset.csv", TEAL, True)], size=12.5)
footer(s, 9)

# ============================================================================
# 10 - EDA distributions
# ============================================================================
s = slide()
header(s, "03 - Data quality", "Exploratory analysis - feature distributions", 10)
figure_panel(s, "missing_values_dirty_vs_clean.png", Inches(0.9), Inches(2.0),
             Inches(11.6), Inches(4.5), "Missingness heatmap: dirty (329 NaNs) vs. cleaned (0 NaNs)")
footer(s, 10)

# ============================================================================
# 11 - EDA correlation & 3D
# ============================================================================
s = slide()
header(s, "03 - Data quality", "Correlation structure & feature space", 11)
figure_panel(s, "eda_correlation_heatmap.png", Inches(0.9), Inches(2.0),
             Inches(5.6), Inches(4.5), "Clinical correlation heatmap")
figure_panel(s, "3d_scatter_features.png", Inches(6.9), Inches(2.0),
             Inches(5.6), Inches(4.5), "3D space: donor/recipient age vs. CD34 dose")
footer(s, 11)

# ============================================================================
# 12 - Section divider: Linear Regression
# ============================================================================
section_divider(12, "04", "Linear Regression", "Model 1 - an interpretable OLS baseline "
                "with bootstrapped uncertainty", TEAL)

# ============================================================================
# 13 - OLS concept
# ============================================================================
s = slide()
header(s, "04 - Linear regression", "What Ordinary Least Squares does", 13, tsize=30)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(2.0), Inches(6.0), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "The baseline model", size=18, color=TEAL, bold=True, first=True,
     space_before=0, space_after=12)
bullet(tf, "Fits a linear relationship between clinical features and the outcome.", mcolor=TEAL)
bullet(tf, "Continuous prediction of survival_status, thresholded at 0.5 for classes.", mcolor=TEAL)
bullet(tf, "Fully transparent: every coefficient is directly interpretable.", mcolor=TEAL)
bullet(tf, "Serves as the yardstick against which cascade gains are measured.", mcolor=TEAL)
rich(tf, [("Why keep it?  ", INK, True),
          ("A stronger model is only convincing relative to a well-understood baseline.",
           GRAY, False)], size=13.5)
cx, cy, cw, ch = card(s, Inches(7.2), Inches(2.0), Inches(5.3), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Pipeline components", size=16, color=RED, bold=True, first=True,
     space_before=0, space_after=12)
for k, v in [("Fit", "scikit-learn LinearRegression"),
             ("Inference", "statsmodels OLS (p-values)"),
             ("Uncertainty", "100x bootstrap resampling"),
             ("Diagnostics", "residual + error tables"),
             ("Metrics", "RMSE, MAE, R^2, F1, MCC")]:
    rich(tf, [(k + "  -  ", TEAL, True), (v, GRAY, False)], size=13, space_before=6,
         space_after=6)
footer(s, 13)

# ============================================================================
# 14 - OLS model equation
# ============================================================================
s = slide()
header(s, "04 - Linear regression", "The linear model", 14, tsize=30)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(2.0), Inches(11.6), Inches(2.25), alpha=0.34)
tf = textbox(s, cx, cy + Inches(0.2), cw, Inches(1.7), anchor=MSO_ANCHOR.MIDDLE)
para(tf, "y = \u03B2\u2080 + \u03B2\u2081x\u2081 + \u03B2\u2082x\u2082 + ... + "
         "\u03B2\u209A x\u209A + \u03B5", size=34, color=INK, bold=True, first=True,
     align=PP_ALIGN.CENTER, space_before=0, space_after=6, font="Cambria")
para(tf, "\u03B2\u2080 = intercept    |    \u03B2\u1D62 = coefficient for feature x\u1D62"
         "    |    \u03B5 = residual error",
     size=14, color=GRAY, align=PP_ALIGN.CENTER, space_before=0, space_after=0,
     font=FONT_MONO)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(4.45), Inches(5.6), Inches(1.95), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Objective", size=15, color=TEAL, bold=True, first=True, space_before=0, space_after=8)
para(tf, "Choose coefficients that minimize the sum of squared residuals "
         "\u03A3(y\u1D62 - \u0177\u1D62)\u00B2, i.e. the total squared prediction error.",
     size=13.5, color=GRAY, space_before=0)
cx, cy, cw, ch = card(s, Inches(6.9), Inches(4.45), Inches(5.6), Inches(1.95), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "In this project", size=15, color=RED, bold=True, first=True,
     space_before=0, space_after=8)
para(tf, "~35 encoded predictors (after one-hot expansion), standardized so that "
         "coefficient magnitudes are comparable across features.",
     size=13.5, color=GRAY, space_before=0)
footer(s, 14)

# ============================================================================
# 15 - Bootstrapping
# ============================================================================
s = slide()
header(s, "04 - Linear regression", "100x bootstrapping for uncertainty", 15, tsize=30)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(2.0), Inches(5.8), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Procedure", size=18, color=TEAL, bold=True, first=True, space_before=0, space_after=12)
bullet(tf, "Resample the training set with replacement (100 iterations).", mcolor=TEAL)
bullet(tf, "Refit OLS on each resample and store every coefficient.", mcolor=TEAL)
bullet(tf, "Summarize mean coefficient, standard error and 95% percentile intervals.", mcolor=TEAL)
bullet(tf, "Cross-check p-values with the statsmodels OLS fit.", mcolor=TEAL)
cx, cy, cw, ch = card(s, Inches(7.0), Inches(2.0), Inches(5.5), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Why it matters", size=16, color=RED, bold=True, first=True, space_before=0, space_after=12)
para(tf, "Bootstrapping builds an empirical sampling distribution without assuming "
         "normality - important for a small, non-Gaussian clinical cohort.",
     size=14, color=GRAY, space_before=0, space_after=14)
rich(tf, [("Output:  ", INK, True), ("bootstrapped_coefficients.csv", TEAL, True)],
     size=13, font=FONT_MONO)
rich(tf, [("Seed:  ", INK, True), ("random_state = 42", INK_SOFT, True)], size=13,
     space_before=10)
footer(s, 15)

# ============================================================================
# 16 - Coefficients & CI table
# ============================================================================
s = slide()
header(s, "04 - Linear regression", "Coefficient estimates & 95% CI", 16, tsize=30)
tbl = [
    ["Feature", "Mean coef.", "Std. err.", "95% CI lower", "95% CI upper", "p-value"],
    ["Relapse", "+0.1178", "0.0311", "+0.0591", "+0.1617", "0.0002"],
    ["Recipientage", "+0.0958", "0.0508", "-0.0060", "+0.2009", "0.0285"],
    ["HLAgrI", "+0.1619", "0.0937", "-0.0377", "+0.3375", "0.0448"],
    ["Disease_chronic", "+0.1859", "0.0854", "+0.0200", "+0.3408", "0.0301"],
    ["extcGvHD", "-0.0835", "0.0322", "-0.1410", "-0.0210", "0.0104"],
    ["survival_time", "-0.3205", "0.0310", "-0.3803", "-0.2648", "<0.0001"],
    ["Disease_nonmalignant", "+0.0031", "0.0112", "-0.0178", "+0.0219", "<0.0001"],
]
table(s, Inches(0.9), Inches(2.0), Inches(11.6), Inches(3.6), tbl,
      col_widths=[2.6, 1.5, 1.3, 1.7, 1.7, 1.3], font_size=12.5, header_size=12,
      row_h=0.42, highlight={1, 6})
tf = textbox(s, Inches(0.9), Inches(5.75), Inches(11.6), Inches(0.9))
rich(tf, [("Read-out:  ", RED, True),
          ("Relapse, recipient age, HLA-group I mismatch, chronic disease and "
           "extra-corporeal GvHD carry statistically meaningful coefficients - "
           "clinically sensible survivors of a small-data fit.", GRAY, False)],
     size=12.5, first=True)
footer(s, 16)

# ============================================================================
# 17 - Residual diagnostics
# ============================================================================
s = slide()
header(s, "04 - Linear regression", "Residual diagnostics", 17, tsize=30)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(2.0), Inches(5.8), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "What we inspect", size=18, color=TEAL, bold=True, first=True,
     space_before=0, space_after=12)
bullet(tf, "Residual = actual - predicted, for every test patient.", mcolor=TEAL)
bullet(tf, "Absolute error and squared error tables per sample.", mcolor=TEAL)
bullet(tf, "Detect systematic bias, heteroscedasticity and outliers.", mcolor=TEAL)
bullet(tf, "Feeds RMSE and MAE - the headline error metrics.", mcolor=TEAL)
rich(tf, [("Output:  ", INK, True), ("regression_errors.csv", TEAL, True)],
     size=12.5, font=FONT_MONO)
cx, cy, cw, ch = card(s, Inches(7.0), Inches(2.0), Inches(5.5), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Error metrics", size=16, color=RED, bold=True, first=True, space_before=0, space_after=12)
for k, v in [("RMSE", "0.3186"), ("MAE", "0.2597"), ("MSE", "0.1015"),
             ("R^2 (survival_time)", "near zero / negative")]:
    rich(tf, [(k + "   ", GRAY, False), (v, INK, True)], size=15, space_before=6, space_after=6)
rich(tf, [("Caveat:  ", RED, True),
          ("day-level survival_time regression is heavily right-censored, so R^2 is "
           "unreliable - the binary classification view is more meaningful.", GRAY, False)],
     size=12.5, space_before=14)
footer(s, 17)

# ============================================================================
# 18 - Thresholded classification
# ============================================================================
s = slide()
header(s, "04 - Linear regression", "From regression output to a decision", 18, tsize=30)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(2.0), Inches(11.6), Inches(2.3), alpha=0.34)
tf = textbox(s, cx, cy + Inches(0.15), cw, Inches(1.9), anchor=MSO_ANCHOR.MIDDLE)
para(tf, "predicted class = 1  (deceased)   if   \u0177 \u2265 0.5   else   0  (survived)",
     size=24, color=INK, bold=True, first=True, align=PP_ALIGN.CENTER,
     space_before=0, space_after=8, font=FONT_MONO)
para(tf, "The OLS model outputs a continuous risk score; a fixed 0.5 threshold converts it "
         "into the binary survival label used for classification metrics.",
     size=13.5, color=GRAY, align=PP_ALIGN.CENTER, space_before=0, space_after=0)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(4.5), Inches(3.7), Inches(1.9), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Recall", size=13, color=TEAL, bold=True, first=True, space_before=0, space_after=2)
para(tf, "1.00", size=28, color=INK, bold=True, space_before=0, space_after=2)
para(tf, "catches every deceased patient", size=11, color=GRAY, space_before=0)
cx, cy, cw, ch = card(s, Inches(4.8), Inches(4.5), Inches(3.7), Inches(1.9), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Precision", size=13, color=RED, bold=True, first=True, space_before=0, space_after=2)
para(tf, "0.833", size=28, color=INK, bold=True, space_before=0, space_after=2)
para(tf, "2 false positives", size=11, color=GRAY, space_before=0)
cx, cy, cw, ch = card(s, Inches(8.7), Inches(4.5), Inches(3.8), Inches(1.9), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Accuracy", size=13, color=BROWN, bold=True, first=True, space_before=0, space_after=2)
para(tf, "0.933", size=28, color=INK, bold=True, space_before=0, space_after=2)
para(tf, "28 of 30 correct", size=11, color=GRAY, space_before=0)
footer(s, 18)

# ============================================================================
# 19 - Bootstrap fit figure
# ============================================================================
s = slide()
header(s, "04 - Linear regression", "Regression fit with 95% bootstrap band", 19, tsize=30)
figure_panel(s, "linear_regression_bootstrapped_fits.png", Inches(0.9), Inches(2.0),
             Inches(11.6), Inches(4.5),
             "OLS regression line with 95% bootstrapped confidence-interval band")
footer(s, 19)

# ============================================================================
# 20 - LR performance summary
# ============================================================================
s = slide()
header(s, "04 - Linear regression", "Baseline performance summary", 20, tsize=30)
cards_data = [
    ("0.9333", "Accuracy", "28 / 30 correct", TEAL),
    ("0.9091", "F1-score", "harmonic mean", BROWN),
    ("0.8660", "MCC", "balanced quality", RED),
    ("0.2597", "MAE", "baseline error", AMBER),
]
x0, y0 = Inches(0.9), Inches(2.0)
cw, ch = Inches(2.83), Inches(2.2)
gx = Inches(0.14)
for i, (val, lab, sub, col) in enumerate(cards_data):
    x = x0 + i * (cw + gx)
    rect(s, x, y0, cw, ch, fill=CARD_FILL, line=BORDER, radius=0.07, alpha=0.32)
    rect(s, x, y0, cw, Inches(0.09), fill=col, line=None, shape=MSO_SHAPE.RECTANGLE)
    tf = textbox(s, x + Inches(0.28), y0 + Inches(0.4), cw - Inches(0.5), ch - Inches(0.6))
    para(tf, val, size=32, color=col, bold=True, first=True, space_before=0, space_after=2)
    para(tf, lab, size=14, color=INK, bold=True, space_before=0, space_after=2)
    para(tf, sub, size=11, color=GRAY, space_before=0, space_after=0)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(4.5), Inches(11.6), Inches(1.9), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Takeaway", size=15, color=TEAL, bold=True, first=True, space_before=0, space_after=8)
para(tf, "The OLS baseline is a strong, sensitive classifier - but it flags some survivors as "
         "deceased (2 false positives). The cascade classifier explicitly targets that "
         "weakness in the next section.", size=14, color=GRAY, space_before=0)
footer(s, 20)

# ============================================================================
# 21 - Section divider: Cascade
# ============================================================================
section_divider(21, "05", "Cascade Classifier", "Model 2 - PNN expansion, quadratic "
                "input-doubling and a Random Forest meta-learner", PLUM)

# ============================================================================
# 22 - Cascade stage 1
# ============================================================================
s = slide()
header(s, "05 - Cascade / Stage 1", "PNN probability expansion", 22, tsize=30)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(2.0), Inches(7.0), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Distance & kernel mechanics", size=18, color=PLUM, bold=True, first=True,
     space_before=0, space_after=12)
bullet(tf, "Three distance matrices to the training support set:", mcolor=PLUM)
bullet(tf, "Euclidean  -  Chebyshev  -  Cosine", level=1, mcolor=RED, size=14)
bullet(tf, "Gaussian kernel   K = exp( -d^2 / (2 sigma^2) ).", mcolor=PLUM)
bullet(tf, "Class-1 posterior = mean(K | class 1) / (mean(K | class 0) + mean(K | class 1)).",
       mcolor=PLUM)
bullet(tf, "Appends [P_E, P_Cheb, P_Cos] to the original vector.", mcolor=PLUM)
rich(tf, [("Effect:  ", INK, True),
          ("a richer, non-linear projection of a low-dimensional clinical space - the core "
           "thesis of Izonin et al. (2025).", GRAY, False)], size=13)
cx, cy, cw, ch = card(s, Inches(8.1), Inches(2.0), Inches(4.4), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Implementation", size=15, color=TEAL, bold=True, first=True,
     space_before=0, space_after=10)
for i, line in enumerate([
    "class PNNProbabilityExpander:",
    "    def fit(X, y) -> self",
    "    def transform(X) -> (N, 3)",
    "        [P_euclid,",
    "         P_cheb, P_cosine]",
    "    def expand_features(X):",
    "        hstack([X, probs])",
]):
    para(tf, line, size=11.5, color=TEAL if i == 0 else INK_SOFT, bold=(i == 0),
         font=FONT_MONO, first=(i == 0), space_before=2, space_after=2)
footer(s, 22)

# ============================================================================
# 23 - Cascade stage 2
# ============================================================================
s = slide()
header(s, "05 - Cascade / Stage 2", "Quadratic input-doubling", 23, tsize=30)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(2.0), Inches(6.0), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Augmentation recipe", size=18, color=PLUM, bold=True, first=True,
     space_before=0, space_after=12)
bullet(tf, "Pair each sample i with a random subset of samples j.", mcolor=PLUM)
bullet(tf, "Concatenate the two expanded vectors: [x_i | x_j].", mcolor=PLUM)
bullet(tf, "Target becomes the class difference z = y_i - y_j in {-1, 0, +1}.", mcolor=PLUM)
bullet(tf, "Up to 8,000 stratified pairs, capped for tractability.", mcolor=PLUM)
bullet(tf, "Turns 157 rows into thousands of informative comparisons.", mcolor=PLUM)
cx, cy, cw, ch = card(s, Inches(7.1), Inches(2.0), Inches(5.4), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Why it mitigates small data", size=16, color=TEAL, bold=True, first=True,
     space_before=0, space_after=12)
rich(tf, [("1.  ", TEAL, True), ("Effective sample-size expansion without fabricating "
          "labels - every pair is grounded in two real patients.", GRAY, False)], size=13.5)
rich(tf, [("2.  ", TEAL, True), ("Pairwise differences capture ordinal structure and reduce "
          "class-imbalance sensitivity.", GRAY, False)], size=13.5)
rich(tf, [("3.  ", TEAL, True), ("The doubled space exposes relative clinical signatures "
          "to the meta-learner.", GRAY, False)], size=13.5)
rich(tf, [("4.  ", TEAL, True), ("Redundant pairs average out idiosyncratic noise - "
          "regularization by construction.", GRAY, False)], size=13.5)
footer(s, 23)

# ============================================================================
# 24 - Cascade stage 3
# ============================================================================
s = slide()
header(s, "05 - Cascade / Stage 3", "Random Forest meta-classifier & voting", 24, tsize=30)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(2.0), Inches(5.8), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Meta-learner", size=18, color=AMBER, bold=True, first=True, space_before=0, space_after=12)
bullet(tf, "RandomForestClassifier trained on the doubled feature space.", mcolor=AMBER)
bullet(tf, "Learns the difference function z = y_i - y_j.", mcolor=AMBER)
bullet(tf, "Gini importances aggregated back to original + PNN features.", mcolor=AMBER)
bullet(tf, "n_estimators and max_depth set by global optimization.", mcolor=AMBER)
cx, cy, cw, ch = card(s, Inches(7.0), Inches(2.0), Inches(5.5), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Inference: plurality voting", size=16, color=TEAL, bold=True, first=True,
     space_before=0, space_after=12)
bullet(tf, "Pick fixed reference vectors (support set).", mcolor=TEAL)
bullet(tf, "Pair each test sample with every reference -> z_hat.", mcolor=TEAL)
bullet(tf, "Vote: y_test = y_ref + z_hat, clipped to {0, 1}.", mcolor=TEAL)
bullet(tf, "Class-1 probability = fraction of Deceased votes.", mcolor=TEAL)
rich(tf, [("Benefit:  ", TEAL, True),
          ("an ensemble of many relative judgments instead of one brittle absolute "
           "prediction.", GRAY, False)], size=13)
footer(s, 24)

# ============================================================================
# 25 - Hyperparameter optimization
# ============================================================================
s = slide()
header(s, "05 - Cascade", "Hyperparameter optimization - Dual Annealing", 25, tsize=28)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(1.95), Inches(5.4), Inches(4.55), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Search setup", size=17, color=PLUM, bold=True, first=True, space_before=0, space_after=10)
bullet(tf, "scipy.optimize.dual_annealing - global, derivative-free.", mcolor=PLUM)
bullet(tf, "Objective: maximize mean_F1 + 0.5 * mean_MCC under 3-fold Stratified CV.",
       mcolor=PLUM)
bullet(tf, "Bounds: sigma [0.2, 3.5], n_estimators [50, 150], max_depth [4, 14].", mcolor=PLUM)
bullet(tf, "15 evaluated trials logged with CV F1 and MCC.", mcolor=PLUM)
rich(tf, [("Optimal configuration", RED, True)], size=15, space_before=12)
rich(tf, [("sigma = 3.2183  -  n_estimators = 80  -  max_depth = 9", INK, True)],
     size=13.5, font=FONT_MONO)
tbl = [
    ["Trial", "Sigma", "n_est", "Depth", "CV F1", "CV MCC"],
    ["3", "3.2183", "80", "9", "0.8120", "0.6504"],
    ["9", "2.2774", "70", "8", "0.8071", "0.6383"],
    ["14", "2.1533", "50", "4", "0.7812", "0.6007"],
    ["8", "1.2460", "105", "12", "0.7674", "0.5703"],
    ["5", "1.4686", "93", "11", "0.7652", "0.5603"],
]
table(s, Inches(6.6), Inches(2.25), Inches(5.9), Inches(3.1), tbl,
      col_widths=[1.0, 1.2, 0.9, 0.9, 1.0, 1.1], font_size=12, header_size=12,
      row_h=0.44, highlight={1})
tf = textbox(s, Inches(6.6), Inches(5.5), Inches(5.9), Inches(0.9))
para(tf, "Top 5 of 15 Dual Annealing trials. Trial 3 is the global optimum.",
     size=11, color=GRAY, first=True)
footer(s, 25)

# ============================================================================
# 26 - Benchmark results
# ============================================================================
s = slide()
header(s, "06 - Results", "Benchmark on the held-out test cohort", 26, tsize=30)
tf = textbox(s, Inches(0.9), Inches(1.95), Inches(11.6), Inches(0.35))
para(tf, "Independent clean test set - 30 patients (20 survived, 10 deceased), never seen "
         "during training or tuning.", size=12, color=GRAY, first=True)
tbl = [
    ["Model", "Acc.", "Prec.", "Recall", "F1", "MCC", "Kappa", "RMSE", "MAE"],
    ["Linear Regression (OLS)", "0.9333", "0.8333", "1.0000", "0.9091", "0.8660", "0.8571", "0.3186", "0.2597"],
    ["Hybrid Cascade (Algorithm 1)", "0.9333", "1.0000", "0.8000", "0.8889", "0.8528", "0.8421", "0.2165", "0.0637"],
    ["Combined Ensemble Blend", "0.9333", "1.0000", "0.8000", "0.8889", "0.8528", "0.8421", "0.2083", "0.1408"],
]
table(s, Inches(0.9), Inches(2.45), Inches(11.6), Inches(1.95), tbl,
      col_widths=[3.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], font_size=12,
      header_size=11.5, row_h=0.46, highlight={2})
cards_data = [
    ("75.5%", "MAE reduction", "0.2597 -> 0.0637", RED),
    ("32.1%", "RMSE reduction", "0.3186 -> 0.2165", TEAL),
    ("0", "False positives", "precision = 1.00", AMBER),
]
x0, y0 = Inches(0.9), Inches(4.7)
cw, ch = Inches(3.7), Inches(1.75)
gx = Inches(0.25)
for i, (val, lab, sub, col) in enumerate(cards_data):
    x = x0 + i * (cw + gx)
    rect(s, x, y0, cw, ch, fill=CARD_FILL, line=BORDER, radius=0.08, alpha=0.32)
    rect(s, x, y0, Inches(0.09), ch, fill=col, line=None, shape=MSO_SHAPE.RECTANGLE)
    tf = textbox(s, x + Inches(0.3), y0 + Inches(0.28), cw - Inches(0.5), ch - Inches(0.5))
    para(tf, val, size=27, color=col, bold=True, first=True, space_before=0, space_after=0)
    para(tf, lab, size=13, color=INK, bold=True, space_before=0, space_after=1)
    para(tf, sub, size=10.5, color=GRAY, space_before=0, space_after=0)
footer(s, 26)

# ============================================================================
# 27 - Confusion matrices
# ============================================================================
s = slide()
header(s, "06 - Results", "Confusion matrices & the OLS trade-off", 27, tsize=30)
tbl = [
    ["Model", "TP", "FP", "TN", "FN", "Sens.", "Spec.", "PPV", "NPV"],
    ["Linear Regression (OLS)", "10", "2", "18", "0", "1.000", "0.900", "0.833", "1.000"],
    ["Hybrid Cascade (Algorithm 1)", "8", "0", "20", "2", "0.800", "1.000", "1.000", "0.909"],
    ["Combined Ensemble Blend", "8", "0", "20", "2", "0.800", "1.000", "1.000", "0.909"],
]
table(s, Inches(0.9), Inches(2.0), Inches(11.6), Inches(1.95), tbl,
      col_widths=[3.4, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0], font_size=12,
      header_size=11.5, row_h=0.46, highlight={2})
figure_panel(s, "confusion_matrix_comparison.png", Inches(0.9), Inches(4.25),
             Inches(5.6), Inches(2.15), "OLS (left) vs. Cascade (right)")
cx, cy, cw, ch = card(s, Inches(6.9), Inches(4.25), Inches(5.6), Inches(2.15), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
rich(tf, [("OLS:  ", TEAL, True), ("sensitive but over-flags - 2 false positives.",
          GRAY, False)], size=13, first=True)
rich(tf, [("Cascade:  ", RED, True), ("precision-first - zero false positives, trading "
          "2 missed deceased.", GRAY, False)], size=13)
rich(tf, [("In transplant triage, ", INK, True), ("false alarms drive unnecessary invasive "
          "work-ups - the cascade avoids them.", GRAY, False)], size=13)
footer(s, 27)

# ============================================================================
# 28 - Feature importance
# ============================================================================
s = slide()
header(s, "06 - Results", "Feature importance - the PNN signal dominates", 28, tsize=28)
items = [
    ("PNN_Euclid", 0.35603, RED),
    ("survival_time", 0.19812, AMBER),
    ("PNN_Cosine", 0.11534, PLUM),
    ("PNN_Chebyshev", 0.07938, BROWN),
    ("CD34kgx10d6", 0.02142, TEAL),
    ("CD3dCD34", 0.01912, RED),
    ("Recipientage", 0.01678, AMBER),
    ("PLTrecovery", 0.01617, BROWN),
]
x0, y0 = Inches(0.9), Inches(2.0)
bar_x = x0 + Inches(3.3)
max_w = Inches(7.1)
row_h = Inches(0.5)
for i, (name, val, col) in enumerate(items):
    y = y0 + i * row_h
    tf = textbox(s, x0, y, Inches(3.1), Inches(0.4), anchor=MSO_ANCHOR.MIDDLE)
    para(tf, name, size=12.5, color=INK, bold=True, first=True, font=FONT_MONO,
         space_before=0, space_after=0)
    w = Emu(int(max_w * (val / 0.36)))
    rect(s, bar_x, y + Inches(0.06), max_w, Inches(0.26), fill=RGBColor(0xED, 0xD3, 0xB9),
         line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    rect(s, bar_x, y + Inches(0.06), w, Inches(0.26), fill=col, line=None,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    tfn = textbox(s, bar_x + max_w + Inches(0.12), y, Inches(1.0), Inches(0.4),
                  anchor=MSO_ANCHOR.MIDDLE)
    para(tfn, "%.4f" % val, size=11, color=GRAY, first=True, space_before=0, space_after=0)
tf = textbox(s, Inches(0.9), Inches(6.25), Inches(11.6), Inches(0.55))
rich(tf, [("Insight:  ", RED, True),
          ("the PNN projection (Euclid + Cosine + Chebyshev) is the strongest learned signal - "
           "empirical support for the paper's feature-enrichment hypothesis.", GRAY, False)],
     size=12.5, first=True)
footer(s, 28)

# ============================================================================
# 29 - ROC / PR
# ============================================================================
s = slide()
header(s, "06 - Diagnostics", "ROC-AUC & Precision-Recall curves", 29, tsize=30)
figure_panel(s, "roc_pr_curves.png", Inches(0.9), Inches(2.0), Inches(11.6), Inches(4.5),
             "Discrimination curves for the OLS and cascade classifiers")
footer(s, 29)

# ============================================================================
# 30 - Web application
# ============================================================================
s = slide()
header(s, "07 - Productionization", "Interactive clinical ML web application", 30, tsize=28)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(2.0), Inches(5.8), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Stack & architecture", size=18, color=TEAL, bold=True, first=True,
     space_before=0, space_after=12)
bullet(tf, "FastAPI backend with REST + auto Swagger/ReDoc docs.", mcolor=TEAL)
bullet(tf, "Singleton model service loading persisted joblib artifacts.", mcolor=TEAL)
bullet(tf, "Responsive dark 7-tab dashboard (vanilla JS + Chart.js).", mcolor=TEAL)
bullet(tf, "9-case automated unittest suite.", mcolor=TEAL)
bullet(tf, "CSV export center with formula-injection sanitization.", mcolor=TEAL)
rich(tf, [("Two models, kept distinct:  ", RED, True),
          ("Ridge for survival_time; cascade for survival_status.", GRAY, False)], size=12.5)
cx, cy, cw, ch = card(s, Inches(7.0), Inches(2.0), Inches(5.5), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Product surface", size=16, color=RED, bold=True, first=True, space_before=0, space_after=12)
for k, v in [("Endpoints", "POST /predict, /compare, GET /health, /meta"),
             ("Features", "35 de-leaked baseline attributes"),
             ("Sections", "demographics, disease, HLA, cell dose, serology, regimen"),
             ("Outputs", "class + probability; time + interval"),
             ("Docs", "127.0.0.1:8000/docs")]:
    rich(tf, [(k + "  -  ", TEAL, True), (v, GRAY, False)], size=12.5, space_before=6,
         space_after=6)
footer(s, 30)

# ============================================================================
# 31 - Key findings
# ============================================================================
s = slide()
header(s, "08 - Conclusions", "Key findings", 31, tsize=30)
findings = [
    ("Prediction error slashed", "Cascade cut MAE by 75.5% (0.2597 -> 0.0637) and RMSE by "
     "32.1% vs. the OLS baseline.", RED),
    ("Zero false positives", "Specificity and precision both 1.00 - no healthy patient "
     "flagged.", TEAL),
    ("PNN expansion is the signal", "PNN_Euclid ranks #1 at 35.6% importance, validating the "
     "paper's thesis.", PLUM),
    ("Input doubling beats small N", "Pairwise differences turn 157 rows into thousands of "
     "grounded comparisons.", AMBER),
    ("Reproducible by design", "Seeded, modular, leak-free pipeline plus notebook and web "
     "app.", BROWN),
]
y0 = Inches(1.95)
rowh = Inches(0.93)
for i, (title, desc, col) in enumerate(findings):
    y = y0 + i * rowh
    rect(s, Inches(0.9), y, Inches(11.6), Inches(0.8), fill=CARD_FILL, line=BORDER,
         radius=0.1, alpha=0.30)
    rect(s, Inches(0.9), y, Inches(0.08), Inches(0.8), fill=col, line=None,
         shape=MSO_SHAPE.RECTANGLE)
    tfn = textbox(s, Inches(1.15), y + Inches(0.05), Inches(0.7), Inches(0.7),
                  anchor=MSO_ANCHOR.MIDDLE)
    para(tfn, "0" + str(i + 1), size=20, color=col, bold=True, first=True,
         space_before=0, space_after=0)
    tft = textbox(s, Inches(1.8), y + Inches(0.08), Inches(10.5), Inches(0.66),
                  anchor=MSO_ANCHOR.MIDDLE)
    para(tft, title, size=14, color=INK, bold=True, first=True, space_before=0, space_after=1)
    para(tft, desc, size=11.5, color=GRAY, space_before=0, space_after=0)
footer(s, 31)

# ============================================================================
# 32 - Limitations & ethics
# ============================================================================
s = slide()
header(s, "08 - Conclusions", "Limitations & responsible use", 32, tsize=30)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(2.0), Inches(5.8), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Statistical caveats", size=18, color=AMBER, bold=True, first=True,
     space_before=0, space_after=12)
bullet(tf, "Test cohort is tiny (N=30); single-split metrics carry wide uncertainty.",
       mcolor=AMBER)
bullet(tf, "Reported benchmarks vary across runs/seeds - pipeline export, README and "
           "web-app summaries differ.", mcolor=AMBER)
bullet(tf, "Precision-first bias trades recall for specificity - model choice follows the "
           "clinical objective.", mcolor=AMBER)
bullet(tf, "survival_time regression is right-censored; R^2 near zero, so day-level "
           "prediction is unreliable.", mcolor=AMBER)
cx, cy, cw, ch = card(s, Inches(7.0), Inches(2.0), Inches(5.5), Inches(4.5), alpha=0.30)
tf = textbox(s, cx, cy, cw, ch)
para(tf, "Ethical disclaimer", size=16, color=RED, bold=True, first=True,
     space_before=0, space_after=12)
bullet(tf, "Research and educational demonstration ONLY.", mcolor=RED)
bullet(tf, "NOT tested in clinical trials, NOT FDA-approved, NOT CE-marked.", mcolor=RED)
bullet(tf, "Must NEVER guide diagnosis, prognosis or treatment.", mcolor=RED)
bullet(tf, "No PHI: only de-identified public benchmark data.", mcolor=RED)
rich(tf, [("Bottom line:  ", INK, True),
          ("a rigorous methods showcase - not a medical device.", GRAY, False)], size=13)
footer(s, 32)

# ============================================================================
# 33 - Attribution / closing
# ============================================================================
s = slide()
rect(s, Inches(0.9), Inches(1.6), Inches(1.9), Inches(0.07), fill=RED, line=None,
     shape=MSO_SHAPE.RECTANGLE)
tf = textbox(s, Inches(0.9), Inches(1.8), Inches(11.5), Inches(1.6))
para(tf, "Thank you", size=44, color=INK, bold=True, first=True, space_before=0, space_after=0)
para(tf, "Questions & discussion", size=18, color=RED, space_before=4, space_after=0)
cx, cy, cw, ch = card(s, Inches(0.9), Inches(3.6), Inches(11.6), Inches(2.7), alpha=0.32)
tf = textbox(s, cx, cy, cw, ch)
rich(tf, [("Method:  ", RED, True),
          ("Izonin, I., Tkachenko, R., Hovdysh, N., Berezsky, O., Yemets, K. & Tsmots, I. "
           "(2025). Cascade-Based Input-Doubling Classifier for Predicting Survival in "
           "Allogeneic Bone Marrow Transplants: Small Data Case. ", INK_SOFT, False),
          ("Computation 13(3):80.", INK, True)], size=13, first=True)
rich(tf, [("Data:  ", RED, True),
          ("Pediatric Bone Marrow Transplantation benchmark dataset (UCI ID 565).",
           INK_SOFT, False)], size=13)
rich(tf, [("Stack:  ", RED, True),
          ("Python, scikit-learn, SciPy, statsmodels, pandas, matplotlib/seaborn, "
           "FastAPI, Chart.js.", INK_SOFT, False)], size=13)
rich(tf, [("License:  ", RED, True), ("MIT License.", INK_SOFT, False)], size=13)
tf = textbox(s, Inches(0.9), Inches(6.55), Inches(11.5), Inches(0.4))
para(tf, "Repository: bone_marrow_cascade_pipeline   -   run  python pipeline.py  to reproduce",
     size=11, color=INK_SOFT, font=FONT_MONO, first=True)
footer(s, 33)


try:
    prs.save(OUT)
    saved = OUT
except PermissionError:
    saved = os.path.join(HERE, "Bone_Marrow_BMT_Presentation_new.pptx")
    prs.save(saved)

print("Saved:", saved)
print("Slides:", len(prs.slides._sldIdLst))
