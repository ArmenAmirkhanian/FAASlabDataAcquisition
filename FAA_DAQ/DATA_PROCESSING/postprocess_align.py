"""
postprocess_align.py — Automatic phase-alignment correction with interactive markers.

Workflow:
  1. File browser — select trimmed input file
  2. Marker plot — click to place Green (good start) / Red (bad start) pairs.
     Right-click or Undo to remove last. Click "Run →" when done.
  3. Alignment runs — manual patches first, then auto-detect forward.
  4. Output review — INPUT (solid+dots) and OUTPUT (dashed+dots) side by side,
     same colour per channel. Red dashed lines = patch locations. Save or Discard.
  5. Save-as dialog.

Patch per event:
  DELETE       DELETE_ROWS rows   (skipped)
  DUPLICATE    DUPLICATE_ROWS rows (written twice)
  KEEP EXTRA   KEEP_EXTRA rows    (good cycles, reference seed)
  Net: +(DUPLICATE_ROWS − DELETE_ROWS) = +8 rows per patch.

Usage:
    python postprocess_align.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16
CYCLE_HZ       = 1.0
RAMP_SECONDS   = 5.0
DELETE_ROWS    = 40
DUPLICATE_ROWS = 48
KEEP_EXTRA     = 48
REF_CYCLES     = 5
BAD_THRESHOLD  = 3

WINDOW_S  = 20.0        # seconds visible in interactive plots
MARKER_MS = 4           # dot size for scatter-line plots
LINE_W    = 0.9         # line width

# ── Derived ───────────────────────────────────────────────────────────────────
SPC       = int(round(SAMPLE_RATE / CYCLE_HZ))
RAMP_ROWS = int(RAMP_SECONDS * SAMPLE_RATE)

# ── 24-colour palette (same index = same channel in both plots) ───────────────
_PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
    "#393b79","#637939","#8c6d31","#843c39",
]

# ── File browser ──────────────────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()
in_path = filedialog.askopenfilename(
    title="Select file to align (*_trimmed.txt)",
    initialdir=os.path.dirname(os.path.abspath(__file__)),
    filetypes=[("Trimmed files", "*_trimmed.txt"),
               ("Text files",    "*.txt"),
               ("All files",     "*.*")]
)
if not in_path:
    messagebox.showinfo("Cancelled", "No file selected.")
    root.destroy()
    raise SystemExit(0)
root.destroy()

# ── Load ──────────────────────────────────────────────────────────────────────
df      = pd.read_csv(in_path, sep="\t")
cols    = [c for c in df.columns if c != "time_s"]
data    = df[cols].values.astype(float)
N, n_ch = data.shape
time_in = np.arange(0, N) / SAMPLE_RATE

det_idx = [i for i, c in enumerate(cols) if c.startswith("DCDT_")]
if not det_idx:
    det_idx = list(range(min(4, n_ch)))

# One colour per channel, consistent across all plots
chan_colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_ch)]

print(f"\nInput      : {os.path.basename(in_path)}")
print(f"Rows       : {N:,}  ({time_in[-1]:.2f} s)")
print(f"Channels   : {n_ch}  |  detection on {len(det_idx)} DCDT channels")
print(f"Patch      : delete {DELETE_ROWS}, duplicate {DUPLICATE_ROWS}, "
      f"keep_extra {KEEP_EXTRA}  (net {DUPLICATE_ROWS - DELETE_ROWS:+d} rows/patch)")
print(f"Threshold  : RMSE > {BAD_THRESHOLD}× median  |  ref cycles : {REF_CYCLES}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — draw one axes with dot-line style
# ══════════════════════════════════════════════════════════════════════════════
def plot_channels(ax, t_arr, d_arr, t0, t1, vis, linestyle="-", title=""):
    mask = (t_arr >= t0) & (t_arr <= t1)
    ax.cla()
    y_labels = []
    for i, c in enumerate(cols):
        if not vis[i]:
            continue
        ax.plot(t_arr[mask], d_arr[mask, i],
                color=chan_colors[i], lw=LINE_W,
                marker="o", markersize=MARKER_MS,
                linestyle=linestyle, label=c, alpha=0.85)
        if c.startswith("DCDT_") and "Displacement" not in y_labels:
            y_labels.append("Displacement")
        elif ("pressure" in c.lower() or c.startswith("volt_ch")) \
                and "Pressure/Voltage" not in y_labels:
            y_labels.append("Pressure/Voltage")
        elif c.startswith("SG_") and "Strain" not in y_labels:
            y_labels.append("Strain")
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("  /  ".join(y_labels) if y_labels else "Value", fontsize=7)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.set_xlim(t0, t1)
    ax.grid(True, alpha=0.25)
    plotted = [cols[i] for i in range(n_ch) if vis[i]]
    if plotted:
        ax.legend(fontsize=6, ncol=2, loc="upper right")


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MARKER SELECTION
# ══════════════════════════════════════════════════════════════════════════════
def select_markers():
    """
    Shows input file with dot-line plot.
    Alternating clicks: Green (good start) → Red (bad start) → Green → …
    Right-click or Undo removes last marker.  Click legend line to toggle channel.
    Returns [(good_start_s, bad_start_s), …] sorted by time.
    """
    markers     = []
    state       = {'next': 'good'}
    leg_line_map = {}
    vis         = [c.startswith("DCDT_") or c.startswith("volt_ch") for c in cols]
    if not any(vis):
        vis = [True] + [False] * (len(cols) - 1)

    fig = plt.figure(figsize=(17, 9))
    plot_ax  = fig.add_axes([0.08, 0.13, 0.88, 0.75])
    slide_ax = fig.add_axes([0.08, 0.04, 0.88, 0.035])

    bw, bh, bx = 0.055, 0.042, 0.004
    btn_dcdt  = Button(fig.add_axes([bx, 0.83, bw, bh]), "DCDT",    color="#d0e8ff", hovercolor="#b0cfff")
    btn_sg    = Button(fig.add_axes([bx, 0.78, bw, bh]), "Strain",  color="#ffd0d0", hovercolor="#ffb0b0")
    btn_press = Button(fig.add_axes([bx, 0.73, bw, bh]), "Pressure",color="#ffe0b0", hovercolor="#ffc870")
    btn_all   = Button(fig.add_axes([bx, 0.68, bw, bh]), "All on",  color="#d0ffd0", hovercolor="#b0ffb0")
    btn_none  = Button(fig.add_axes([bx, 0.63, bw, bh]), "All off", color="#e8e8e8", hovercolor="#d0d0d0")
    btn_undo  = Button(fig.add_axes([bx, 0.55, bw, bh]), "Undo",    color="#fff0b0", hovercolor="#ffe070")
    btn_next  = Button(fig.add_axes([bx, 0.50, bw, bh]), "Next\n20s",color="#dde8ff", hovercolor="#bbd0ff")
    btn_prev  = Button(fig.add_axes([bx, 0.44, bw, bh]), "Prev\n20s",color="#dde8ff", hovercolor="#bbd0ff")
    btn_run   = Button(fig.add_axes([bx, 0.38, bw, bh]), "Run →",   color="#90ee90", hovercolor="#60dd60")
    for b in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none, btn_undo, btn_next, btn_prev, btn_run):
        b.label.set_fontsize(7)

    # Live marker-time readout below the buttons
    info_ax  = fig.add_axes([bx, 0.06, bw, 0.29])
    info_ax.axis("off")
    info_txt = info_ax.text(0.05, 0.98, "No markers\nplaced yet",
                             transform=info_ax.transAxes,
                             fontsize=6.5, va="top", ha="left",
                             family="monospace", color="#222222")

    t_max  = max(float(time_in[-1]) - WINDOW_S, 0.01)
    slider = Slider(slide_ax, "Time (s)", 0.0, t_max, valinit=0.0, color="steelblue")

    def title_str():
        nxt = state['next']
        n_g = sum(1 for k, _ in markers if k == 'good')
        n_b = sum(1 for k, _ in markers if k == 'bad')
        return (
            f"{os.path.basename(in_path)}  —  "
            f"next click → {'GREEN (good start)' if nxt == 'good' else 'RED (bad start)'}  |  "
            f"{len(markers)} markers  ({min(n_g, n_b)} pairs)  |  "
            f"right-click = undo  •  click legend to toggle  •  'Run →' when done"
        )

    def redraw(_=None):
        t0 = slider.val
        t1 = t0 + WINDOW_S
        plot_channels(plot_ax, time_in, data, t0, t1, vis,
                      linestyle="-", title="INPUT — place Good/Bad markers")

        # Wire legend lines for click-to-toggle
        leg = plot_ax.get_legend()
        leg_line_map.clear()
        if leg:
            vis_idx = [i for i in range(n_ch) if vis[i]]
            for j, ll in enumerate(leg.get_lines()):
                if j < len(vis_idx):
                    ll.set_picker(8)
                    ll.set_linewidth(2.5)
                    leg_line_map[ll] = vis_idx[j]

        ylim  = plot_ax.get_ylim()
        yspan = max(ylim[1] - ylim[0], 1e-9)
        for kind, t in markers:
            clr = "#2ca02c" if kind == 'good' else "#d62728"
            lbl = "G" if kind == 'good' else "B"
            plot_ax.axvline(t, color=clr, lw=2.0, alpha=0.85, zorder=5)
            plot_ax.text(t, ylim[0] + yspan * 0.92, lbl,
                         color=clr, fontsize=9, fontweight="bold",
                         ha="center", zorder=6)

        fig.suptitle(title_str(), fontsize=8.5, fontweight="bold")

        # Update live marker-time readout
        lines   = []
        g_count = b_count = 0
        for kind, t in markers:
            if kind == 'good':
                g_count += 1
                lines.append(f"G{g_count}: {t:.2f} s")
            else:
                b_count += 1
                lines.append(f"B{b_count}: {t:.2f} s")
        info_txt.set_text("\n".join(lines) if lines else "No markers\nplaced yet")

        fig.canvas.draw_idle()

    def set_group(group):
        for i, c in enumerate(cols):
            if   group == "dcdt":  vis[i] = c.startswith("DCDT_")
            elif group == "sg":    vis[i] = c.startswith("SG_")
            elif group == "press": vis[i] = "pressure" in c.lower() or c.startswith("volt_ch")
            elif group == "all":   vis[i] = True
            elif group == "none":  vis[i] = False
        redraw()

    def on_legend_pick(event):
        ll = event.artist
        if ll not in leg_line_map:
            return
        vis[leg_line_map[ll]] = not vis[leg_line_map[ll]]
        redraw()

    def on_click(event):
        if event.inaxes != plot_ax:
            return
        if event.xdata is None:
            return
        if event.button == 3:
            do_undo(None)
            return
        if event.button == 1:
            # Snap to nearest data point in time_in
            idx = int(np.argmin(np.abs(time_in - event.xdata)))
            t   = float(time_in[idx])
            markers.append((state['next'], t))
            state['next'] = 'bad' if state['next'] == 'good' else 'good'
            redraw()

    def do_undo(_):
        if markers:
            markers.pop()
            state['next'] = 'good' if state['next'] == 'bad' else 'bad'
            redraw()

    def do_next(_):
        slider.set_val(min(slider.val + 20.0, t_max))

    def do_prev(_):
        slider.set_val(max(slider.val - 20.0, 0.0))

    def do_run(_):
        plt.close(fig)

    slider.on_changed(redraw)
    btn_dcdt.on_clicked(lambda _:  set_group("dcdt"))
    btn_sg.on_clicked(lambda _:    set_group("sg"))
    btn_press.on_clicked(lambda _: set_group("press"))
    btn_all.on_clicked(lambda _:   set_group("all"))
    btn_none.on_clicked(lambda _:  set_group("none"))
    btn_undo.on_clicked(do_undo)
    btn_next.on_clicked(do_next)
    btn_prev.on_clicked(do_prev)
    btn_run.on_clicked(do_run)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("pick_event", on_legend_pick)

    redraw()
    plt.show()

    good_times = sorted(t for k, t in markers if k == 'good')
    bad_times  = sorted(t for k, t in markers if k == 'bad')
    return list(zip(good_times, bad_times))


# ══════════════════════════════════════════════════════════════════════════════
# ALIGNMENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
output_chunks  = []
patches        = []
patch_out_rows = []
out_row        = 0

def do_patch(patch_row):
    global out_row
    dup_start  = min(patch_row + DELETE_ROWS, N)
    dup_end    = min(dup_start + DUPLICATE_ROWS, N)
    keep_start = dup_end
    keep_end   = min(keep_start + KEEP_EXTRA, N)

    dup_block  = data[dup_start:dup_end]
    keep_block = data[keep_start:keep_end]

    patch_out_rows.append(out_row)
    output_chunks.append(dup_block)
    output_chunks.append(dup_block)
    output_chunks.append(keep_block)
    out_row += len(dup_block) * 2 + len(keep_block)

    new_out  = np.vstack([dup_block, dup_block, keep_block])
    ref_rows = new_out[-REF_CYCLES * SPC:] if len(new_out) >= REF_CYCLES * SPC else new_out
    return keep_end, ref_rows


def compute_ref(ref_rows):
    n_cyc = len(ref_rows) // SPC
    if n_cyc < 1:
        return None, 0.0
    seg     = ref_rows[:n_cyc * SPC].reshape(n_cyc, SPC, n_ch)
    det     = seg[:, :, det_idx]
    ref     = np.median(det, axis=0)
    rmse_ea = np.sqrt(np.mean((det - ref[np.newaxis])**2, axis=(1, 2)))
    return ref, float(np.median(rmse_ea))


# ══════════════════════════════════════════════════════════════════════════════
# RUN MARKER SELECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\nOpening marker selection plot...")
print("  Left-click  : Green (good start) then Red (bad start), alternating")
print("  Right-click : undo last marker")
print("  Run →       : proceed when all visible bad cycles are marked\n")

manual_patches = select_markers()
print(f"\n  {len(manual_patches)} manual patch(es) from markers")
print()
print(f"  {'#':>4}  {'Type':>8}  {'Input row':>10}  {'Time (s)':>10}  {'Net':>5}")
print(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*5}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: MANUAL PATCHES
# ══════════════════════════════════════════════════════════════════════════════
input_pos = 0
ref_rows  = None

for k, (good_start_s, bad_start_s) in enumerate(manual_patches):
    bad_row = int(round(bad_start_s * SAMPLE_RATE))
    if bad_row <= input_pos or bad_row >= N:
        print(f"  WARNING: patch {k+1} bad_start_s={bad_start_s:.2f} s out of range — skipping.")
        continue

    chunk = data[input_pos:bad_row]
    output_chunks.append(chunk)
    out_row += len(chunk)

    input_pos, ref_rows = do_patch(bad_row)
    patches.append(("manual", bad_row))
    print(f"  {len(patches):>4}  {'manual':>8}  {bad_row:>10,}  "
          f"{bad_start_s:>10.4f}  {DUPLICATE_ROWS - DELETE_ROWS:>+5d}")


# ══════════════════════════════════════════════════════════════════════════════
# SEED REFERENCE IF NO MANUAL PATCHES
# ══════════════════════════════════════════════════════════════════════════════
if input_pos == 0:
    seed_end = min(RAMP_ROWS + REF_CYCLES * SPC, N)
    output_chunks.append(data[:seed_end])
    ref_rows  = data[RAMP_ROWS:seed_end]
    input_pos = seed_end
    out_row   = seed_end

if ref_rows is None or len(ref_rows) < SPC:
    print("\nERROR: not enough reference data. Place at least one marker pair.")
    raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: AUTO-DETECT
# ══════════════════════════════════════════════════════════════════════════════
reference, med_rmse = compute_ref(ref_rows)
clean_pos = input_pos
scan_pos  = input_pos

while scan_pos + SPC <= N:
    block      = data[scan_pos:scan_pos + SPC, :][:, det_idx]
    block_rmse = float(np.sqrt(np.mean((block - reference) ** 2)))
    ratio      = block_rmse / med_rmse if med_rmse > 0 else 0.0
    print(f"  t={scan_pos/SAMPLE_RATE:7.2f}s  rmse={block_rmse:.5f}  ratio={ratio:.3f}")

    if med_rmse > 0 and block_rmse > BAD_THRESHOLD * med_rmse:
        # Look ahead: if the NEXT cycle is also bad, don't go back 1 SPC —
        # patch from the start of the first bad cycle to preserve the good cycle before it.
        if scan_pos + 2 * SPC <= N:
            nxt_block = data[scan_pos + SPC:scan_pos + 2 * SPC, :][:, det_idx]
            nxt_rmse  = float(np.sqrt(np.mean((nxt_block - reference) ** 2)))
            two_bad   = nxt_rmse > BAD_THRESHOLD * med_rmse
        else:
            two_bad = False

        if two_bad:
            patch_row  = max(clean_pos, scan_pos)       # 2 consecutive bad: start at first bad
            patch_type = "auto-2x"
        else:
            patch_row  = max(clean_pos, scan_pos - SPC) # 1 bad: go back 1 good cycle
            patch_type = "auto"

        chunk = data[clean_pos:patch_row]
        if len(chunk):
            output_chunks.append(chunk)
            out_row += len(chunk)

        input_pos, ref_rows = do_patch(patch_row)
        reference, med_rmse = compute_ref(ref_rows)
        clean_pos = input_pos
        scan_pos  = input_pos

        patches.append((patch_type, patch_row))
        print(f"  {len(patches):>4}  {patch_type:>8}  {patch_row:>10,}  "
              f"{patch_row / SAMPLE_RATE:>10.4f}  {DUPLICATE_ROWS - DELETE_ROWS:>+5d}")
    else:
        scan_pos += SPC

if clean_pos < N:
    output_chunks.append(data[clean_pos:N])
    out_row += N - clean_pos


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
output_data     = np.vstack(output_chunks)
M               = len(output_data)
new_time        = np.arange(0, M) / SAMPLE_RATE
patch_out_times = [r / SAMPLE_RATE for r in patch_out_rows]

n_manual  = sum(1 for p in patches if p[0] == "manual")
n_auto    = sum(1 for p in patches if p[0] == "auto")
n_auto2x  = sum(1 for p in patches if p[0] == "auto-2x")

print(f"\n{'─'*54}")
if not patches:
    print("  No bad cycles detected.")
print(f"  Patches     : {len(patches):,}  ({n_manual} manual, {n_auto} auto-1x, {n_auto2x} auto-2x)")
print(f"  Input rows  : {N:,}  ({time_in[-1]:.4f} s)")
print(f"  Output rows : {M:,}  ({new_time[-1]:.4f} s)")
print(f"  Net change  : {M - N:+,} rows")
print(f"{'─'*54}")


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT REVIEW PLOT — single axes, INPUT (solid+dots) + OUTPUT (dashed+dots)
# ══════════════════════════════════════════════════════════════════════════════
def show_output_review():
    """Single overlaid plot: input solid, output dashed, same colour per channel.
    Click legend line to toggle that channel. Returns True = Save, False = Discard."""
    state        = {'decision': None}
    leg_line_map = {}
    vis          = [c.startswith("DCDT_") or c.startswith("volt_ch") for c in cols]
    if not any(vis):
        vis = [True] + [False] * (len(cols) - 1)

    fig      = plt.figure(figsize=(17, 9))
    plot_ax  = fig.add_axes([0.08, 0.13, 0.88, 0.75])
    slide_ax = fig.add_axes([0.08, 0.04, 0.88, 0.035])

    bw, bh, bx = 0.055, 0.042, 0.004
    btn_dcdt    = Button(fig.add_axes([bx, 0.83, bw, bh]), "DCDT",    color="#d0e8ff", hovercolor="#b0cfff")
    btn_sg      = Button(fig.add_axes([bx, 0.78, bw, bh]), "Strain",  color="#ffd0d0", hovercolor="#ffb0b0")
    btn_press   = Button(fig.add_axes([bx, 0.73, bw, bh]), "Pressure",color="#ffe0b0", hovercolor="#ffc870")
    btn_all     = Button(fig.add_axes([bx, 0.68, bw, bh]), "All on",  color="#d0ffd0", hovercolor="#b0ffb0")
    btn_none    = Button(fig.add_axes([bx, 0.63, bw, bh]), "All off", color="#e8e8e8", hovercolor="#d0d0d0")
    btn_next    = Button(fig.add_axes([bx, 0.55, bw, bh]), "Next\n20s",color="#dde8ff", hovercolor="#bbd0ff")
    btn_prev    = Button(fig.add_axes([bx, 0.50, bw, bh]), "Prev\n20s",color="#dde8ff", hovercolor="#bbd0ff")
    btn_save    = Button(fig.add_axes([bx, 0.44, bw, bh]), "Save",    color="#90ee90", hovercolor="#60dd60")
    btn_discard = Button(fig.add_axes([bx, 0.39, bw, bh]), "Discard", color="#ffaaaa", hovercolor="#ff8888")
    for b in (btn_dcdt, btn_sg, btn_press, btn_all, btn_none, btn_next, btn_prev, btn_save, btn_discard):
        b.label.set_fontsize(7)

    t_data_max = max(float(time_in[-1]), float(new_time[-1]))
    t_max      = max(t_data_max - WINDOW_S, 0.01)
    slider     = Slider(slide_ax, "Time (s)", 0.0, t_max, valinit=0.0, color="steelblue")

    fig.suptitle(
        f"OUTPUT REVIEW — {os.path.basename(in_path)}  |  "
        f"{len(patches)} patches ({n_manual} manual + {n_auto} auto-1x + {n_auto2x} auto-2x)  |  "
        f"Input {N:,} rows → Output {M:,} rows  ({M - N:+,})\n"
        "——  INPUT (original)     - - -  OUTPUT (fixed)     "
        "same colour = same channel     red dashed = patch location     "
        "click legend to toggle channel",
        fontsize=8.5, fontweight="bold"
    )

    def redraw(_=None):
        t0       = slider.val
        t1       = t0 + WINDOW_S
        mask_in  = (time_in  >= t0) & (time_in  <= t1)
        mask_out = (new_time >= t0) & (new_time <= t1)

        plot_ax.cla()
        y_labels  = []
        vis_idx   = [i for i in range(n_ch) if vis[i]]

        for i in vis_idx:
            c   = cols[i]
            clr = chan_colors[i]
            plot_ax.plot(time_in[mask_in],   data[mask_in, i],
                         color=clr, lw=LINE_W, linestyle="-",
                         marker="o", markersize=MARKER_MS,
                         alpha=0.90, label=f"{c} (in)")
            plot_ax.plot(new_time[mask_out], output_data[mask_out, i],
                         color=clr, lw=LINE_W, linestyle="--",
                         marker="o", markersize=MARKER_MS,
                         alpha=0.65, label=f"{c} (out)")
            if c.startswith("DCDT_") and "Displacement" not in y_labels:
                y_labels.append("Displacement")
            elif ("pressure" in c.lower() or c.startswith("volt_ch")) \
                    and "Pressure/Voltage" not in y_labels:
                y_labels.append("Pressure/Voltage")
            elif c.startswith("SG_") and "Strain" not in y_labels:
                y_labels.append("Strain")

        for pt in patch_out_times:
            if t0 - 1 <= pt <= t1 + 1:
                plot_ax.axvline(pt, color="red", lw=1.4, ls="--", alpha=0.65, zorder=3)

        plot_ax.set_xlabel("Time (s)", fontsize=9)
        plot_ax.set_ylabel("  /  ".join(y_labels) if y_labels else "Value", fontsize=8)
        plot_ax.set_xlim(t0, t1)
        plot_ax.grid(True, alpha=0.25)

        # Build legend and wire click-to-toggle
        # Each visible channel has 2 legend entries: (in) and (out)
        leg_line_map.clear()
        if vis_idx:
            leg      = plot_ax.legend(fontsize=6, ncol=3, loc="upper right")
            leg_lines = leg.get_lines()
            for j, chan_i in enumerate(vis_idx):
                for k in range(2):          # 0 = (in), 1 = (out)
                    ll_idx = j * 2 + k
                    if ll_idx < len(leg_lines):
                        leg_lines[ll_idx].set_picker(8)
                        leg_lines[ll_idx].set_linewidth(2.5)
                        leg_line_map[leg_lines[ll_idx]] = chan_i

        fig.canvas.draw_idle()

    def set_group(group):
        for i, c in enumerate(cols):
            if   group == "dcdt":  vis[i] = c.startswith("DCDT_")
            elif group == "sg":    vis[i] = c.startswith("SG_")
            elif group == "press": vis[i] = "pressure" in c.lower() or c.startswith("volt_ch")
            elif group == "all":   vis[i] = True
            elif group == "none":  vis[i] = False
        redraw()

    def on_legend_pick(event):
        ll = event.artist
        if ll not in leg_line_map:
            return
        vis[leg_line_map[ll]] = not vis[leg_line_map[ll]]
        redraw()

    def do_next(_):
        slider.set_val(min(slider.val + 20.0, t_max))

    def do_prev(_):
        slider.set_val(max(slider.val - 20.0, 0.0))

    def on_save(_):
        state['decision'] = True
        plt.close(fig)

    def on_discard(_):
        state['decision'] = False
        plt.close(fig)

    slider.on_changed(redraw)
    btn_dcdt.on_clicked(lambda _:  set_group("dcdt"))
    btn_sg.on_clicked(lambda _:    set_group("sg"))
    btn_press.on_clicked(lambda _: set_group("press"))
    btn_all.on_clicked(lambda _:   set_group("all"))
    btn_none.on_clicked(lambda _:  set_group("none"))
    btn_next.on_clicked(do_next)
    btn_prev.on_clicked(do_prev)
    btn_save.on_clicked(on_save)
    btn_discard.on_clicked(on_discard)
    fig.canvas.mpl_connect("pick_event", on_legend_pick)

    redraw()
    plt.show()
    return state['decision']


print("\nOpening output review plot...")
save_it = show_output_review()

if not save_it:
    print("\nDiscarded — no file written.")
    raise SystemExit(0)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE-AS DIALOG + WRITE
# ══════════════════════════════════════════════════════════════════════════════
default_name = os.path.splitext(os.path.basename(in_path))[0] + "_aligned.txt"
save_root = tk.Tk()
save_root.withdraw()
save_dir = filedialog.askdirectory(
    title="Select folder to save aligned file",
    initialdir=os.path.dirname(in_path)
)
save_root.destroy()
if not save_dir:
    print("\nSave cancelled — no file written.")
    raise SystemExit(0)
out_path = os.path.join(save_dir, default_name)

print(f"\nWriting: {out_path}")
with open(out_path, "w") as f:
    f.write("time_s\t" + "\t".join(cols) + "\n")
    for i in range(M):
        row_vals = [f"{new_time[i]:.6f}"] + [f"{v:.6f}" for v in output_data[i]]
        f.write("\t".join(row_vals) + "\n")

print(f"Done — {M:,} rows written.")
print(f"  Time : {new_time[0]:.4f} s → {new_time[-1]:.4f} s")
