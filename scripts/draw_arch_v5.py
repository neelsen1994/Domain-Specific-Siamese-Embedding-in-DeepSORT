#!/usr/bin/env python3
"""
Publication-quality architecture diagrams for Model v5 (Turkey ReID).
Designed for IEEE Access journal (vector-safe, Type-1 fonts, 300 DPI PNG).

Outputs (created in ./figures/):
    arch_v5_pipeline.svg   — full forward-pass pipeline (vector, for papers)
    arch_v5_pipeline.pdf   — same, PDF vector
    arch_v5_pipeline.png   — same, 300 DPI raster
    arch_v5_resblock.svg   — Pre-Act ResBlock + SE detail (vector, for papers)
    arch_v5_resblock.pdf   — same, PDF vector
    arch_v5_resblock.png   — same, 300 DPI raster

Usage:
    conda run -n mot python scripts/draw_arch_v5.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

os.makedirs("figures", exist_ok=True)

# ── IEEE-compatible font settings ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "pdf.fonttype":       42,           # Type-1 TrueType (required by IEEE)
    "ps.fonttype":        42,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.bottom": False,
    "axes.spines.left":   False,
    "xtick.bottom":       False,
    "ytick.left":         False,
})

# ── Color palette: distinguishable in color AND grayscale ─────────────────────
# Each entry: (facecolor, edgecolor)
PAL = {
    "input":   ("#D6EAF8", "#2471A3"),   # light blue  / blue
    "conv":    ("#D5E8D4", "#196F3D"),   # light green / dark green
    "pool":    ("#FEF9E7", "#B7950B"),   # light yellow/ dark gold
    "resblk":  ("#EDE0F5", "#6C3483"),   # lavender    / purple
    "norm":    ("#E8F8F5", "#148F77"),   # mint        / teal
    "gap":     ("#D6EAF8", "#1A5276"),   # light blue  / dark blue
    "fc":      ("#FDEBD0", "#935116"),   # peach       / dark orange
    "dropout": ("#F2F3F4", "#566573"),   # light gray  / gray
    "output":  ("#154360", "#154360"),   # solid dark blue (white text)
    "se":      ("#FADBD8", "#922B21"),   # light rose  / dark red
    "skip":    ("#FDFEFE", "#839192"),   # white       / gray
}


# ── Primitive helpers ─────────────────────────────────────────────────────────

def rblock(ax, cx, cy, w, h, kind, title, sub="", dim="", dim_x_offset=0.45, lw=1.3):
    """Draw a rounded-corner rectangle with centred text.

    dim_x_offset: horizontal gap between block right edge and dimension label.
                  Caller should set this large enough to clear any enclosing
                  stage box border.
    """
    fc, ec = PAL[kind]
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(rect)
    tc = "white" if kind == "output" else "#111111"
    dy = 0.14 * h if sub else 0.0
    ax.text(cx, cy + dy, title,
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=tc, zorder=4)
    if sub:
        ax.text(cx, cy - 0.30 * h, sub,
                ha="center", va="center",
                fontsize=7.5, color=tc, style="italic", zorder=4)
    if dim:
        ax.text(cx + w / 2 + dim_x_offset, cy, dim,
                ha="left", va="center",
                fontsize=7.5, color="#5D6D7E",
                fontfamily="monospace", zorder=4)


def down_arrow(ax, x, y_from, y_to, color="#2C3E50", mut=9):
    ax.annotate("",
                xy=(x, y_to + 0.03), xytext=(x, y_from - 0.03),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.1, mutation_scale=mut),
                zorder=5)


def right_arrow(ax, x_from, x_to, y, color="#2C3E50", mut=9):
    ax.annotate("",
                xy=(x_to - 0.03, y), xytext=(x_from + 0.03, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.1, mutation_scale=mut),
                zorder=5)


def stage_box(ax, cx, bw, y_top, y_bot, label, col="#7F8C8D"):
    """Dashed bounding box around a stage with a rotated label on the left."""
    pad = 0.20
    rect = FancyBboxPatch(
        (cx - bw / 2 - pad, y_bot - pad),
        bw + 2 * pad, (y_top - y_bot) + 2 * pad,
        boxstyle="round,pad=0.06",
        facecolor="#FAFAFA", edgecolor=col,
        linewidth=0.9, linestyle="--", zorder=1,
    )
    ax.add_patch(rect)
    # Place label clearly to the left of the stage box
    ax.text(cx - bw / 2 - pad - 0.22,
            (y_top + y_bot) / 2,
            label,
            ha="right", va="center",
            fontsize=8.5, fontweight="bold",
            color=col, rotation=90)


# =============================================================================
# Figure 1 — Full pipeline
# =============================================================================

def draw_pipeline() -> None:
    # ── Layout constants ──────────────────────────────────────────────────────
    CX   = 5.5          # block centre x  (shifted right to leave room for stage labels)
    BW   = 6.8          # block width
    SH   = 0.70         # standard block height
    TH   = 0.80         # tall block height (×2 repeat blocks)
    G_IN = 0.75         # gap between stages
    G_WI = 0.30         # gap within a stage

    # Stage box geometry: right edge = CX + BW/2 + pad = 5.5 + 3.4 + 0.20 = 9.10
    # dim text starts at CX + BW/2 + DIM_OFF = 5.5 + 3.4 + 0.45 = 9.35  → outside box ✓
    DIM_OFF = 0.45

    # ── Block definitions (kind, title, subtitle, dim_label, height) ──────────
    # Index 0: Input
    # Indices 1-3: Stage 1  (normalize + conv×2 + maxpool)
    # Indices 4-5: Stage 2  (resblock×2 + final BN/ReLU)
    # Indices 6-7: Stage 3  (conv128 + maxpool)
    # Indices 8-9: Stage 4  (resblock×2 + final BN/ReLU)
    # Indices 10-13: Head   (GAP + BNNeck + Dropout + FC)
    # Index 14: Output
    DEFS = [
        # kind       title                                    sub                         dim             h
        ("input",   "Input Patch",                           "128 × 64 × 3",              "",            SH),
        # --- Stage 1 ---
        ("norm",    "Normalize  (÷ 255)",                    "",                          "128×64×3",    SH),
        ("conv",    "Conv 3×3  (32 ch)  +  BN  +  ReLU",    "×2   |   same padding",     "128×64×32",   TH),
        ("pool",    "MaxPool 3×3   /   stride 2",            "",                          "64×32×32",    SH),
        # --- Stage 2 ---
        ("resblk",  "Pre-Act ResBlock + SE  (32 ch)",        "×2",                        "64×32×32",    TH),
        ("norm",    "BN  +  ReLU",                           "stage-2 finalisation",      "64×32×32",    SH),
        # --- Stage 3 ---
        ("conv",    "Conv 3×3  (128 ch)  +  BN  +  ReLU",   "same padding",              "64×32×128",   SH),
        ("pool",    "MaxPool 3×3   /   stride 2",            "",                          "32×16×128",   SH),
        # --- Stage 4 ---
        ("resblk",  "Pre-Act ResBlock + SE  (128 ch)",       "×2",                        "32×16×128",   TH),
        ("norm",    "BN  +  ReLU",                           "stage-4 finalisation",      "32×16×128",   SH),
        # --- Head ---
        ("gap",     "Global Average Pool",                   "",                          "128-D",       SH),
        ("norm",    "BN  (BNNeck)",                          "",                          "128-D",       SH),
        ("dropout", "Dropout  (p = 0.3)",                    "",                          "128-D",       SH),
        ("fc",      "FC 128  (no bias)  +  L2-Norm",         "",                          "128-D",       SH),
        # --- Output ---
        ("output",  "128-D Embedding",                       "‖ f ‖₂  =  1",             "",            0.85),
    ]

    # Indices at which a new stage begins (G_IN gap applied before these)
    stage_boundaries = {1, 4, 6, 8, 10}

    GROUPS = [
        ("Stage 1",  1,  3,  "conv"),
        ("Stage 2",  4,  5,  "resblk"),
        ("Stage 3",  6,  7,  "conv"),
        ("Stage 4",  8,  9,  "resblk"),
        ("Head",    10, 13,  "gap"),
    ]

    # ── Compute y-centres top→bottom ──────────────────────────────────────────
    yc = []
    y  = 20.5
    yc.append(y)
    for i in range(1, len(DEFS)):
        gap = G_IN if i in stage_boundaries else G_WI
        y = yc[-1] - DEFS[i - 1][4] / 2 - gap - DEFS[i][4] / 2
        yc.append(y)

    # ── Figure ────────────────────────────────────────────────────────────────
    y_top_fig = yc[0]  + DEFS[0][4]   / 2 + 0.35  # no title — small top margin
    y_bot_fig = yc[-1] - DEFS[-1][4]  / 2 - 0.40  # small bottom margin
    fig_h = (y_top_fig - y_bot_fig) * 0.46         # data→inches scale
    fig, ax = plt.subplots(figsize=(5.5, fig_h))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(y_bot_fig, y_top_fig)
    ax.set_aspect("auto")
    ax.axis("off")

    # ── Stage boxes (drawn first, behind blocks) ───────────────────────────────
    for lbl, i0, i1, col_key in GROUPS:
        yt = yc[i0] + DEFS[i0][4] / 2
        yb = yc[i1] - DEFS[i1][4] / 2
        stage_box(ax, CX, BW, yt, yb, lbl, col=PAL[col_key][1])

    # ── Blocks ────────────────────────────────────────────────────────────────
    for i, (kind, title, sub, dim, h) in enumerate(DEFS):
        rblock(ax, CX, yc[i], BW, h, kind, title, sub, dim,
               dim_x_offset=DIM_OFF)

    # ── Arrows ────────────────────────────────────────────────────────────────
    for i in range(len(DEFS) - 1):
        y_from = yc[i]     - DEFS[i][4]     / 2
        y_to   = yc[i + 1] + DEFS[i + 1][4] / 2
        down_arrow(ax, CX, y_from, y_to)

    # ── Dimension column header ───────────────────────────────────────────────
    # Placed to the right of the first block that carries a dim label (index 1)
    hdr_x = CX + BW / 2 + DIM_OFF
    hdr_y = yc[1] + DEFS[1][4] / 2 + 0.28
    ax.text(hdr_x, hdr_y,
            "Feature size\n(H × W × C)",
            ha="left", va="bottom",
            fontsize=7.5, color="#5D6D7E", style="italic")

    plt.tight_layout(pad=0.3)
    for fmt in ("svg", "pdf", "png"):
        out = "figures/arch_v5_pipeline.{}".format(fmt)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print("[saved]", out)
    plt.close()


# =============================================================================
# Figure 2 — Pre-Activation ResBlock + SE detail
# =============================================================================

def draw_resblock() -> None:
    """
    Layout (horizontal, left→right):

     x ──────────────────────────────────────────────────── Add ─── SE ─── x'
      │                                                       ↑
      └─── BN ─► ReLU ─► Conv(3×3) ─► BN ─► ReLU ─► Conv(3×3) ──┘

    SE detail (inset box, below):
      x' ─► GAP ─► FC(C/4, ReLU) ─► FC(C, σ) ─► Reshape ─► ⊗ x' ─► output
    """
    _, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    BH    = 0.72        # block height
    BW_sm = 1.15        # small block (BN, ReLU, Add, SE)
    BW_md = 1.60        # medium block (Conv)
    BW_lg = 1.45        # large block (FC in SE path)
    Y_MAIN = 6.3        # y-centre of main residual path
    Y_SE   = 2.5        # y-centre of SE detail row

    # ── Helper ────────────────────────────────────────────────────────────────
    def next_cx(prev_cx, prev_w, cur_w, gap=0.38):
        return prev_cx + prev_w / 2 + gap + cur_w / 2

    # ── Main path ─────────────────────────────────────────────────────────────
    x_in = 0.85
    ax.plot(x_in, Y_MAIN, "o", color="#2C3E50", ms=7, zorder=5)
    ax.text(x_in, Y_MAIN + 0.62, "x  (input)",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

    path = [
        ("BN",        "norm",  BW_sm),
        ("ReLU",      "norm",  BW_sm),
        ("Conv 3×3",  "conv",  BW_md),
        ("BN",        "norm",  BW_sm),
        ("ReLU",      "norm",  BW_sm),
        ("Conv 3×3",  "conv",  BW_md),
        ("Add  (+)",  "skip",  BW_sm),
        ("SE",        "se",    BW_sm),
    ]

    main_cx = []
    cx = next_cx(x_in, 0, path[0][2], gap=0.52)
    for i, (lbl, kind, w) in enumerate(path):
        main_cx.append(cx)
        rblock(ax, cx, Y_MAIN, w, BH, kind, lbl)
        if i < len(path) - 1:
            cx = next_cx(cx, w, path[i + 1][2], gap=0.33)

    # Shortcut line (arc above main path)
    skip_y = Y_MAIN + BH / 2 + 0.42
    x_add  = main_cx[-2]   # centre of "Add (+)" block
    # vertical segment: input dot up to skip level
    ax.plot([x_in, x_in], [Y_MAIN, skip_y], color="#839192", lw=1.2, zorder=2)
    ax.annotate("",
                xy=(x_add, skip_y), xytext=(x_in, skip_y),
                arrowprops=dict(arrowstyle="-", color="#839192", lw=1.2))
    ax.annotate("",
                xy=(x_add, Y_MAIN + BH / 2 + 0.06),
                xytext=(x_add, skip_y),
                arrowprops=dict(arrowstyle="-|>", color="#839192",
                                lw=1.2, mutation_scale=9))
    ax.text((x_in + x_add) / 2, skip_y + 0.16,
            "shortcut  (identity)",
            ha="center", va="bottom", fontsize=8, color="#839192",
            style="italic")

    # Arrow: input dot → first block
    ax.annotate("",
                xy=(main_cx[0] - path[0][2] / 2 - 0.06, Y_MAIN),
                xytext=(x_in + 0.07, Y_MAIN),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                lw=1.1, mutation_scale=9))

    # Arrows between blocks
    for i in range(len(main_cx) - 1):
        x_a = main_cx[i]     + path[i][2]     / 2
        x_b = main_cx[i + 1] - path[i + 1][2] / 2
        ax.annotate("",
                    xy=(x_b - 0.06, Y_MAIN), xytext=(x_a + 0.06, Y_MAIN),
                    arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                    lw=1.1, mutation_scale=9))

    # Label the Add → SE edge: clarify what value enters SE
    x_add_right = main_cx[-2] + path[-2][2] / 2   # right edge of Add block
    x_se_left   = main_cx[-1] - path[-1][2] / 2   # left  edge of SE  block
    ax.text((x_add_right + x_se_left) / 2, Y_MAIN - BH / 2 - 0.12,
            "Add output",
            ha="center", va="top", fontsize=7.5, color="#2C3E50", style="italic")

    # Output arrow + label
    x_out = main_cx[-1] + path[-1][2] / 2 + 0.45
    ax.annotate("",
                xy=(x_out, Y_MAIN),
                xytext=(main_cx[-1] + path[-1][2] / 2 + 0.08, Y_MAIN),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                lw=1.1, mutation_scale=9))
    ax.text(x_out + 0.10, Y_MAIN, "x'  (output)",
            ha="left", va="center", fontsize=9, fontweight="bold")

    # Channel annotation
    ax.text(x_in, Y_MAIN - BH / 2 - 0.30,
            "C = 32  (Stage 2)   or   128  (Stage 4)",
            ha="left", va="top", fontsize=8, color="#5D6D7E", style="italic")

    # ── SE detail box ─────────────────────────────────────────────────────────
    se_box_x0, se_box_x1 = 0.30, 14.50
    se_box_y0, se_box_y1 = 0.90, 3.90
    ax.add_patch(FancyBboxPatch(
        (se_box_x0, se_box_y0),
        se_box_x1 - se_box_x0, se_box_y1 - se_box_y0,
        boxstyle="round,pad=0.1",
        facecolor="#FEF9E7", edgecolor=PAL["se"][1],
        linewidth=1.0, linestyle="--", zorder=1,
    ))
    ax.text((se_box_x0 + se_box_x1) / 2, se_box_y1 - 0.10,
            "Squeeze-and-Excite (SE) Module — detail",
            ha="center", va="top",
            fontsize=9, fontweight="bold", color=PAL["se"][1])

    # SE path blocks
    se_path = [
        ("GAP\n(squeeze)",    "gap", BW_sm + 0.15),
        ("FC  C/4\n(ReLU)",   "fc",  BW_lg),
        ("FC  C\n(Sigmoid)",  "fc",  BW_lg),
        ("Reshape\n(1×1×C)",  "norm",BW_lg),
        ("⊗  channel\nscale", "se",  BW_lg),
    ]
    x_se_in = 1.10
    ax.plot(x_se_in, Y_SE, "o", color=PAL["se"][1], ms=7, zorder=5)
    ax.text(x_se_in, Y_SE + 0.58, "Add output",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    se_cx = []
    cx = next_cx(x_se_in, 0, se_path[0][2], gap=0.55)
    for i, (lbl, kind, w) in enumerate(se_path):
        se_cx.append(cx)
        rblock(ax, cx, Y_SE, w, BH, kind, lbl)
        if i < len(se_path) - 1:
            cx = next_cx(cx, w, se_path[i + 1][2], gap=0.33)

    # Arrow: SE input → first SE block
    ax.annotate("",
                xy=(se_cx[0] - se_path[0][2] / 2 - 0.06, Y_SE),
                xytext=(x_se_in + 0.07, Y_SE),
                arrowprops=dict(arrowstyle="-|>", color=PAL["se"][1],
                                lw=1.1, mutation_scale=9))

    # Arrows between SE blocks
    for i in range(len(se_cx) - 1):
        x_a = se_cx[i]     + se_path[i][2]     / 2
        x_b = se_cx[i + 1] - se_path[i + 1][2] / 2
        ax.annotate("",
                    xy=(x_b - 0.06, Y_SE), xytext=(x_a + 0.06, Y_SE),
                    arrowprops=dict(arrowstyle="-|>", color=PAL["se"][1],
                                    lw=1.1, mutation_scale=9))

    # Identity fork: x' → down → across → up into ⊗ block
    x_mult  = se_cx[-1]
    fork_y  = Y_SE - BH / 2 - 0.42
    ax.plot([x_se_in, x_se_in], [Y_SE, fork_y],  color=PAL["se"][1], lw=1.2)
    ax.plot([x_se_in, x_mult],  [fork_y, fork_y], color=PAL["se"][1], lw=1.2)
    ax.annotate("",
                xy=(x_mult, Y_SE - BH / 2 + 0.06),
                xytext=(x_mult, fork_y),
                arrowprops=dict(arrowstyle="-|>", color=PAL["se"][1],
                                lw=1.2, mutation_scale=9))
    ax.text((x_se_in + x_mult) / 2, fork_y - 0.12,
            "Add output  passed to  channel-wise multiply",
            ha="center", va="top", fontsize=7.5, color=PAL["se"][1],
            style="italic")

    # SE output arrow + label
    x_se_out = se_cx[-1] + se_path[-1][2] / 2 + 0.45
    ax.annotate("",
                xy=(x_se_out, Y_SE),
                xytext=(se_cx[-1] + se_path[-1][2] / 2 + 0.08, Y_SE),
                arrowprops=dict(arrowstyle="-|>", color=PAL["se"][1],
                                lw=1.1, mutation_scale=9))
    ax.text(x_se_out + 0.10, Y_SE, "output",
            ha="left", va="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout(pad=0.3)
    for fmt in ("svg", "pdf", "png"):
        out = "figures/arch_v5_resblock.{}".format(fmt)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print("[saved]", out)
    plt.close()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("Drawing pipeline diagram ...")
    draw_pipeline()
    print("Drawing ResBlock + SE detail ...")
    draw_resblock()
    print("Done.  Files are in ./figures/")
