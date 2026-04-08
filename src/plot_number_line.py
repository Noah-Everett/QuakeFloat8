"""
Plot representable numbers across the number line for QF8 vs FP8 E4M3.
Shows how each format "embeds" its codepoints on the real line.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── LaTeX-style rendering for paper figures ──────────────────────────────────
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

# ─── QF8 representable values (per-element, no block scaling) ─────────────────
# 8 bits: 1 sign + 7-bit code in u3.4 fixed-point log
# Value = (-1)^s × 2^((code - 64) / 16)
# code=0 → zero; codes 1..127 → positive magnitudes

BIAS = 64
qf8_magnitudes = [0.0]
for code in range(1, 128):
    qf8_magnitudes.append(2.0 ** ((code - BIAS) / 16.0))
qf8_magnitudes = np.array(qf8_magnitudes)

# Full QF8 set: positive + negative
qf8_values = np.sort(np.concatenate([-qf8_magnitudes[1:], qf8_magnitudes]))


# ─── FP8 E4M3 representable values ───────────────────────────────────────────
# 8 bits: 1 sign + 4 exponent + 3 mantissa
# Exponent bias = 7
# Normal:   (-1)^s × 2^(e-7) × (1 + m/8),  e in [1,15], m in [0,7]
# Subnormal: (-1)^s × 2^(-6) × (m/8),       e=0, m in [1,7]  (m=0 → zero)
# E4M3FN (OCP spec): only e=15, m=7 is NaN (no inf). e=15, m=0..6 are valid normals.
# Max value = 2^8 × (1 + 6/8) = 448

fp8_magnitudes = [0.0]

# Subnormals: e=0, m=1..7
for m in range(1, 8):
    fp8_magnitudes.append(2.0**(-6) * (m / 8.0))

# Normals: e=1..15, m=0..7, excluding e=15 m=7 (NaN)
for e in range(1, 16):
    for m in range(8):
        if e == 15 and m == 7:
            continue  # NaN
        fp8_magnitudes.append(2.0**(e - 7) * (1.0 + m / 8.0))

fp8_magnitudes = np.array(sorted(set(fp8_magnitudes)))
fp8_values = np.sort(np.concatenate([-fp8_magnitudes[1:], fp8_magnitudes]))


# ─── FP8 E5M2 representable values ───────────────────────────────────────────
# 8 bits: 1 sign + 5 exponent + 2 mantissa
# Exponent bias = 15
# Normal:   (-1)^s × 2^(e-15) × (1 + m/4),  e in [1,30], m in [0,3]
# Subnormal: (-1)^s × 2^(-14) × (m/4),       e=0, m in [1,3]  (m=0 → zero)
# e=31: m=0 → inf, m=1..3 → NaN

fp8e5m2_magnitudes = [0.0]

# Subnormals: e=0, m=1..3
for m in range(1, 4):
    fp8e5m2_magnitudes.append(2.0**(-14) * (m / 4.0))

# Normals: e=1..30, m=0..3
for e in range(1, 31):
    for m in range(4):
        fp8e5m2_magnitudes.append(2.0**(e - 15) * (1.0 + m / 4.0))

# e=31: inf/NaN (skip)

fp8e5m2_magnitudes = np.array(sorted(set(fp8e5m2_magnitudes)))
fp8e5m2_values = np.sort(np.concatenate([-fp8e5m2_magnitudes[1:], fp8e5m2_magnitudes]))


# ─── INT8 representable values (uniform) ─────────────────────────────────────
# Sign-magnitude: 1 sign + 7-bit code → codes 0..127
# With E8M0 block scaling: value = scale × code
# Unscaled positive magnitudes: {1, 2, ..., 127}  (code=0 → zero)

int8_magnitudes = np.arange(0, 128, dtype=np.float64)  # 0, 1, 2, ..., 127
int8_values_pos = np.linspace(0, qf8_magnitudes[-1], 128)  # scaled version for early plots
int8_values = np.sort(np.concatenate([-int8_values_pos[1:], int8_values_pos]))


# ─── Plot 1: [0, 2] overview ──────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(14, 6), sharex=False)

formats = [
    ("INT8 (uniform)", int8_values, "#888888"),
    ("FP8 E5M2", fp8e5m2_values, "#e67e22"),
    ("FP8 E4M3", fp8_values, "#e74c3c"),
    ("QF8 (log-domain)", qf8_values, "#2980b9"),
]

x_max = 2.0

for ax, (name, values, color) in zip(axes, formats):
    vis = values[(values >= 0) & (values <= x_max)]
    ax.vlines(vis, 0, 1, colors=color, linewidth=0.5, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.02, x_max)
    ax.set_yticks([])
    ax.set_ylabel(name, fontsize=11, rotation=0, ha='right', va='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    n = len(vis)
    ax.text(x_max * 0.98, 0.8, f'{n} values in [0, {x_max}]',
            ha='right', va='top', fontsize=9, color=color, fontweight='bold')

axes[-1].set_xlabel('Value', fontsize=11)
axes[0].set_title('Representable Numbers on the Number Line (positive half, [0, 2])',
                   fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('repo/paper/figures/number_line_embedding.pdf', bbox_inches='tight')
plt.savefig('repo/paper/figures/number_line_embedding.png', bbox_inches='tight', dpi=200)
print("Saved to repo/paper/figures/number_line_embedding.{pdf,png}")
plt.close(fig)

# ─── Paper figures ────────────────────────────────────────────────────────────

paper_formats = [
    (r'\textsc{Int8}', int8_values, '#7f8c8d'),
    (r'\textsc{Fp8} E5M2', fp8e5m2_values, '#e67e22'),
    (r'\textsc{Fp8} E4M3', fp8_values, '#c0392b'),
    (r'\textbf{QF8}', qf8_values, '#2471a3'),
]
n_fmt = len(paper_formats)

# (a) Linear scale [0, 2]
fig_pa, axes_pa = plt.subplots(n_fmt, 1, figsize=(6.5, 2.6), sharex=True)

x_max_pa = 2.0
for ax, (name, values, color) in zip(axes_pa, paper_formats):
    vis = values[(values >= 0) & (values <= x_max_pa)]
    ax.vlines(vis, 0, 1, colors=color, linewidth=0.6, alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.01, x_max_pa)
    ax.set_yticks([])
    ax.set_ylabel(name, fontsize=9, rotation=0, ha='right', va='center',
                  color=color, labelpad=8)
    ax.tick_params(axis='x', labelsize=8, direction='out', length=3, width=0.5)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)
    n = len(vis)
    ax.text(0.99, 0.75, rf'{n} values',
            ha='right', va='top', fontsize=7, color=color,
            transform=ax.transAxes)

axes_pa[-1].set_xlabel(r'Value', fontsize=9)
axes_pa[0].set_title(r'(a) Linear scale, positive half $[0,\,2]$',
                      fontsize=10, pad=6)
fig_pa.subplots_adjust(hspace=0.15)
plt.savefig('repo/paper/fig_number_line_linear.pdf', bbox_inches='tight', pad_inches=0.03)
plt.savefig('repo/paper/figures/fig_number_line_linear.png', bbox_inches='tight', dpi=300)
print("Saved paper figure: fig_number_line_linear.pdf")
plt.close(fig_pa)

# (b) Log₂ scale, full positive range
fig_pb, axes_pb = plt.subplots(n_fmt, 1, figsize=(6.5, 2.6), sharex=True)

for ax, (name, values, color) in zip(axes_pb, paper_formats):
    pos = values[values > 0]
    ax.vlines(pos, 0, 1, colors=color, linewidth=0.4, alpha=0.7)
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(name, fontsize=9, rotation=0, ha='right', va='center',
                  color=color, labelpad=8)
    ax.tick_params(axis='x', labelsize=8, direction='out', length=3, width=0.5)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)
    n = len(pos)
    ax.text(0.99, 0.75, rf'{n} values',
            ha='right', va='top', fontsize=7, color=color,
            transform=ax.transAxes)

axes_pb[-1].set_xlabel(r'Value ($\log_2$ scale)', fontsize=9)
axes_pb[0].set_title(r'(b) $\log_2$ scale, full positive range',
                      fontsize=10, pad=6)
fig_pb.subplots_adjust(hspace=0.15)
plt.savefig('repo/paper/fig_number_line_log.pdf', bbox_inches='tight', pad_inches=0.03)
plt.savefig('repo/paper/figures/fig_number_line_log.png', bbox_inches='tight', dpi=300)
print("Saved paper figure: fig_number_line_log.pdf")
plt.close(fig_pb)

# ─── Plot 2: near-zero zoom ──────────────────────────────────────────────────

fig_zoom, axes_zoom = plt.subplots(4, 1, figsize=(14, 6), sharex=True)

x_max_zoom = 0.15

for ax, (name, values, color) in zip(axes_zoom, formats):
    vis = values[(values >= 0) & (values <= x_max_zoom)]
    ax.vlines(vis, 0, 1, colors=color, linewidth=1.5, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.002, x_max_zoom)
    ax.set_yticks([])
    ax.set_ylabel(name, fontsize=11, rotation=0, ha='right', va='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    n = len(vis)
    ax.text(x_max_zoom * 0.98, 0.8, f'{n} values in [0, {x_max_zoom}]',
            ha='right', va='top', fontsize=9, color=color, fontweight='bold')

axes_zoom[-1].set_xlabel('Value', fontsize=11)
axes_zoom[0].set_title('Representable Numbers Near Zero (positive half, [0, 0.15])',
                        fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('repo/paper/figures/number_line_near_zero.pdf', bbox_inches='tight')
plt.savefig('repo/paper/figures/number_line_near_zero.png', bbox_inches='tight', dpi=200)
print("Saved to repo/paper/figures/number_line_near_zero.{pdf,png}")
plt.close(fig_zoom)

# Also make a log-scale version showing the full positive range
fig2, axes2 = plt.subplots(4, 1, figsize=(14, 6), sharex=True)

for ax, (name, values, color) in zip(axes2, formats):
    pos = values[values > 0]
    ax.vlines(pos, 0, 1, colors=color, linewidth=0.3, alpha=0.6)
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(name, fontsize=11, rotation=0, ha='right', va='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    n = len(pos)
    ax.text(0.98, 0.8, f'{n} positive values',
            ha='right', va='top', fontsize=9, color=color, fontweight='bold',
            transform=ax.transAxes)

axes2[-1].set_xlabel(r'Value ($\log_2$ scale)', fontsize=11)
axes2[0].set_title(r'Representable Numbers --- Full Positive Range ($\log_2$ scale)',
                    fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('repo/paper/figures/number_line_log_scale.pdf', bbox_inches='tight')
plt.savefig('repo/paper/figures/number_line_log_scale.png', bbox_inches='tight', dpi=200)
print("Saved to repo/paper/figures/number_line_log_scale.{pdf,png}")
plt.close(fig2)

# ─── Plot 4: block scaling comparison ────────────────────────────────────────
# Show how E8M0 block scaling shifts the representable window for QF8 vs FP8.
# Each row is one (format, scale) combination, plotted on a shared log₂ axis.

# Unscaled per-element magnitudes (positive only, excluding zero)
qf8_pos = qf8_magnitudes[qf8_magnitudes > 0]
fp8_pos = fp8_magnitudes[fp8_magnitudes > 0]
fp8e5m2_pos = fp8e5m2_magnitudes[fp8e5m2_magnitudes > 0]
int8_pos = int8_magnitudes[int8_magnitudes > 0]  # {1, 2, ..., 127}

scale_exponents = [-4, -2, 0, 2, 4]
n_scales = len(scale_exponents)

fig_block, axes_block = plt.subplots(
    n_scales, 2, figsize=(14, 1.3 * n_scales + 1.5),
    sharex=True, sharey=True,
)

for row, exp in enumerate(scale_exponents):
    scale = 2.0 ** exp

    # FP8 column
    ax_fp8 = axes_block[row, 0]
    vals = fp8_pos * scale
    ax_fp8.vlines(vals, 0, 1, colors='#e74c3c', linewidth=0.4, alpha=0.7)
    ax_fp8.set_ylim(0, 1)
    ax_fp8.set_yticks([])
    ax_fp8.set_xscale('log', base=2)
    ax_fp8.text(0.02, 0.7, rf'scale $= 2^{{{exp}}}$', fontsize=8,
                transform=ax_fp8.transAxes, color='#555')
    for spine in ['top', 'right', 'left']:
        ax_fp8.spines[spine].set_visible(False)

    # QF8 column
    ax_qf8 = axes_block[row, 1]
    vals = qf8_pos * scale
    ax_qf8.vlines(vals, 0, 1, colors='#2980b9', linewidth=0.4, alpha=0.7)
    ax_qf8.set_ylim(0, 1)
    ax_qf8.set_yticks([])
    ax_qf8.set_xscale('log', base=2)
    ax_qf8.text(0.02, 0.7, rf'scale $= 2^{{{exp}}}$', fontsize=8,
                transform=ax_qf8.transAxes, color='#555')
    for spine in ['top', 'right', 'left']:
        ax_qf8.spines[spine].set_visible(False)

axes_block[0, 0].set_title('FP8 E4M3 + E8M0 block scale', fontsize=11,
                            fontweight='bold', color='#e74c3c')
axes_block[0, 1].set_title('QF8 + E8M0 block scale', fontsize=11,
                            fontweight='bold', color='#2980b9')
axes_block[-1, 0].set_xlabel(r'Value ($\log_2$ scale)', fontsize=10)
axes_block[-1, 1].set_xlabel(r'Value ($\log_2$ scale)', fontsize=10)

fig_block.suptitle('Effect of E8M0 Block Scaling on Representable Numbers',
                    fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('repo/paper/figures/number_line_block_scaling.pdf', bbox_inches='tight')
plt.savefig('repo/paper/figures/number_line_block_scaling.png', bbox_inches='tight', dpi=200)
print("Saved to repo/paper/figures/number_line_block_scaling.{pdf,png}")
plt.close(fig_block)

# ─── Plot 5: block scaling comparison (linear scale) ─────────────────────────
# Each row uses its own x-range so the values are visible at each scale.

fig_lin, axes_lin = plt.subplots(
    n_scales, 2, figsize=(14, 1.3 * n_scales + 1.5),
    sharey=True,
)

for row, exp in enumerate(scale_exponents):
    scale = 2.0 ** exp

    fp8_scaled = fp8_pos * scale
    qf8_scaled = qf8_pos * scale

    # Normalize both to [0, 1] so we compare spacing pattern, not range
    fp8_norm = fp8_scaled / fp8_scaled.max()
    qf8_norm = qf8_scaled / qf8_scaled.max()

    # FP8 column
    ax_fp8 = axes_lin[row, 0]
    ax_fp8.vlines(fp8_norm, 0, 1, colors='#e74c3c', linewidth=0.4, alpha=0.7)
    ax_fp8.set_ylim(0, 1)
    ax_fp8.set_xlim(-0.02, 1.02)
    ax_fp8.set_yticks([])
    ax_fp8.text(0.02, 0.7, rf'scale $= 2^{{{exp}}}$ (max $= {fp8_scaled.max():.1f}$)',
                fontsize=8, transform=ax_fp8.transAxes, color='#555')
    for spine in ['top', 'right', 'left']:
        ax_fp8.spines[spine].set_visible(False)

    # QF8 column
    ax_qf8 = axes_lin[row, 1]
    ax_qf8.vlines(qf8_norm, 0, 1, colors='#2980b9', linewidth=0.4, alpha=0.7)
    ax_qf8.set_ylim(0, 1)
    ax_qf8.set_xlim(-0.02, 1.02)
    ax_qf8.set_yticks([])
    ax_qf8.text(0.02, 0.7, rf'scale $= 2^{{{exp}}}$ (max $= {qf8_scaled.max():.1f}$)',
                fontsize=8, transform=ax_qf8.transAxes, color='#555')
    for spine in ['top', 'right', 'left']:
        ax_qf8.spines[spine].set_visible(False)

axes_lin[0, 0].set_title('FP8 E4M3 + E8M0 block scale', fontsize=11,
                           fontweight='bold', color='#e74c3c')
axes_lin[0, 1].set_title('QF8 + E8M0 block scale', fontsize=11,
                           fontweight='bold', color='#2980b9')
axes_lin[-1, 0].set_xlabel('Fraction of max representable value', fontsize=10)
axes_lin[-1, 1].set_xlabel('Fraction of max representable value', fontsize=10)

fig_lin.suptitle('Effect of E8M0 Block Scaling on Representable Numbers (linear scale)',
                  fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('repo/paper/figures/number_line_block_scaling_linear.pdf', bbox_inches='tight')
plt.savefig('repo/paper/figures/number_line_block_scaling_linear.png', bbox_inches='tight', dpi=200)
print("Saved to repo/paper/figures/number_line_block_scaling_linear.{pdf,png}")
plt.close(fig_lin)

# ─── Plot 6: FP8 vs QF8 overlaid, [0, 128], all contributing block scales ───
# One row per E8M0 exponent. Both formats on same axis so you can compare
# exactly which values each format can represent in the [0, 128] window.

x_lo, x_hi = 0.0, 128.0

# Find all integer exponents where at least one value from any format
# falls in [x_lo, x_hi] (excluding zero).
contributing_exps = []
for exp in range(-20, 21):
    scale = 2.0 ** exp
    any_in = False
    for pos in [fp8_pos, fp8e5m2_pos, qf8_pos, int8_pos]:
        vals = pos * scale
        if np.any((vals > x_lo) & (vals <= x_hi)):
            any_in = True
            break
    if any_in:
        contributing_exps.append(exp)

n_rows = len(contributing_exps)
fig_overlay, axes_ov = plt.subplots(
    n_rows, 1, figsize=(14, 0.7 * n_rows + 1.5), sharex=True,
)
if n_rows == 1:
    axes_ov = [axes_ov]

for ax, exp in zip(axes_ov, contributing_exps):
    scale = 2.0 ** exp

    # 4 formats, each in its own vertical band
    overlay_formats = [
        (int8_pos, '#888888', 0.0, 0.22),
        (fp8e5m2_pos, '#e67e22', 0.26, 0.48),
        (fp8_pos, '#e74c3c', 0.52, 0.74),
        (qf8_pos, '#2980b9', 0.78, 1.0),
    ]
    for pos, color, y_lo, y_hi in overlay_formats:
        vals = pos * scale
        vis = vals[(vals > x_lo) & (vals <= x_hi)]
        ax.vlines(vis, y_lo, y_hi, colors=color, linewidth=0.5, alpha=0.8)

    ax.set_ylim(0, 1)
    ax.set_xlim(-1, x_hi + 1)
    ax.set_yticks([])
    ax.text(-0.01, 0.5, rf'$2^{{{exp}}}$', fontsize=8, ha='right', va='center',
            transform=ax.transAxes, color='#555', family='monospace')
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

axes_ov[-1].set_xlabel('Value', fontsize=11)

# Legend at top
for label, color, y in [('INT8', '#888888', 0.11), ('FP8 E5M2', '#e67e22', 0.37),
                         ('FP8 E4M3', '#e74c3c', 0.63), ('QF8', '#2980b9', 0.89)]:
    axes_ov[0].text(0.99, y, label, fontsize=8, color=color,
                    fontweight='bold', ha='right', va='center',
                    transform=axes_ov[0].transAxes)

fig_overlay.suptitle('Representable Numbers in [0, 128] by E8M0 Block Scale',
                      fontsize=13, fontweight='bold')
fig_overlay.text(0.005, 0.5, 'Block scale exponent', fontsize=10, rotation=90,
                 va='center', ha='left', color='#555')
plt.tight_layout(rect=[0.03, 0, 1, 0.97])
plt.savefig('repo/paper/figures/number_line_overlay_0_128.pdf', bbox_inches='tight')
plt.savefig('repo/paper/figures/number_line_overlay_0_128.png', bbox_inches='tight', dpi=200)
print("Saved to repo/paper/figures/number_line_overlay_0_128.{pdf,png}")
plt.close(fig_overlay)

# ─── Plot 7: all block scales smashed onto one line, [0, 128] ────────────────
# Union of all representable values across all contributing E8M0 scales.

# Collect unique values in [0, 128] across all scales for each format
smash_formats = [
    ('INT8', int8_pos, '#888888'),
    ('FP8 E5M2', fp8e5m2_pos, '#e67e22'),
    ('FP8 E4M3', fp8_pos, '#e74c3c'),
    ('QF8', qf8_pos, '#2980b9'),
]

smash_sets = {}
for name, pos, color in smash_formats:
    vals = set()
    for exp in contributing_exps:
        scale = 2.0 ** exp
        for v in pos * scale:
            if x_lo < v <= x_hi:
                vals.add(v)
    smash_sets[name] = np.array(sorted(vals))

n_smash = len(smash_formats)
fig_smash, axes_smash = plt.subplots(n_smash, 1, figsize=(14, 1.5 * n_smash), sharex=True)

for ax, (name, pos, color) in zip(axes_smash, smash_formats):
    vals = smash_sets[name]
    ax.vlines(vals, 0, 1, colors=color, linewidth=0.15, alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(-1, x_hi + 1)
    ax.set_yticks([])
    ax.set_ylabel(name, fontsize=11, rotation=0, ha='right', va='center',
                  color=color, fontweight='bold')
    ax.text(0.99, 0.85, f'{len(vals)} unique values',
            fontsize=9, color=color, fontweight='bold',
            ha='right', va='top', transform=ax.transAxes)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

axes_smash[-1].set_xlabel('Value', fontsize=11)
fig_smash.suptitle('All Representable Numbers in $[0, 128]$ --- Union Across All E8M0 Scales',
                    fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('repo/paper/figures/number_line_smashed_0_128.pdf', bbox_inches='tight')
plt.savefig('repo/paper/figures/number_line_smashed_0_128.png', bbox_inches='tight', dpi=200)
print("Saved to repo/paper/figures/number_line_smashed_0_128.{pdf,png}")
plt.close(fig_smash)

# ─── Plot 8: smashed union, log scale ─────────────────────────────────────────

# Global x-range: min of all formats' smallest value, max = x_hi
global_min = min(smash_sets[name].min() for name, _, _ in smash_formats if len(smash_sets[name]) > 0)

fig_smash_log, axes_smash_log = plt.subplots(n_smash, 1, figsize=(14, 1.5 * n_smash), sharex=True)

for ax, (name, pos, color) in zip(axes_smash_log, smash_formats):
    vals = smash_sets[name]
    ax.vlines(vals, 0, 1, colors=color, linewidth=0.15, alpha=0.5)
    ax.set_xscale('log', base=2)
    ax.set_xlim(global_min * 0.8, x_hi * 1.05)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(name, fontsize=11, rotation=0, ha='right', va='center',
                  color=color, fontweight='bold')
    ax.text(0.99, 0.85, f'{len(vals)} unique values',
            fontsize=9, color=color, fontweight='bold',
            ha='right', va='top', transform=ax.transAxes)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

axes_smash_log[-1].set_xlabel(r'Value ($\log_2$ scale)', fontsize=11)
fig_smash_log.suptitle(r'All Representable Numbers in $[0, 128]$ --- Union Across All E8M0 Scales ($\log_2$)',
                        fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('repo/paper/figures/number_line_smashed_0_128_log.pdf', bbox_inches='tight')
plt.savefig('repo/paper/figures/number_line_smashed_0_128_log.png', bbox_inches='tight', dpi=200)
print("Saved to repo/paper/figures/number_line_smashed_0_128_log.{pdf,png}")
plt.close(fig_smash_log)
