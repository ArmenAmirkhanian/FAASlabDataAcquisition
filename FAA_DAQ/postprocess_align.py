"""
postprocess_align.py — Automatic phase-alignment correction.

Works in sliding windows of WINDOW_CYCLES cycles. Within each window:
  1. Segments data into 16-sample blocks
  2. Overlays all cycles and computes median reference
  3. Flags the first cycle whose RMSE > BAD_THRESHOLD × median RMSE
  4. Patches: deletes DELETE_ROWS rows at the bad cycle start,
     duplicates the next DUPLICATE_ROWS rows, keeps the originals

After each patch the data is back in alignment, so the next window
detects correctly without accumulated phase drift.

Input : any trimmed txt file (tab-separated, time_s column)
Output: <name>_aligned.txt with rebuilt time_s from 1/SAMPLE_RATE

Usage:
    python postprocess_align.py
"""

import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16      # Hz
CYCLE_HZ      = 1.0     # loading frequency
RAMP_SECONDS  = 5.0     # seconds of ramp at start — skipped during detection
WINDOW_CYCLES  = 15      # cycles per detection window
BAD_THRESHOLD  = 2.5     # flag if RMSE > BAD_THRESHOLD × median RMSE in window
DELETE_ROWS    = 40      # rows removed at the bad cycle start
DUPLICATE_ROWS = 48      # rows duplicated immediately after deletion

# ── Derived ───────────────────────────────────────────────────────────────────
SPC       = int(round(SAMPLE_RATE / CYCLE_HZ))   # samples per cycle (16)
RAMP_ROWS = int(RAMP_SECONDS * SAMPLE_RATE)       # rows to skip (80)

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

# Fix time: row i (0-indexed) → t = (i+1) / SAMPLE_RATE
time_fixed = np.arange(0, N) / SAMPLE_RATE

# Use DCDT channels for phase detection — clearest cyclic signal
det_idx = [i for i, c in enumerate(cols) if c.startswith("DCDT_")]
if not det_idx:
    det_idx = list(range(min(4, n_ch)))   # fallback: first 4 channels

print(f"\nInput      : {os.path.basename(in_path)}")
print(f"Rows       : {N:,}  ({time_fixed[-1]:.2f} s)")
print(f"Channels   : {n_ch}  |  detection on {len(det_idx)} DCDT channels")
print(f"Ramp skip  : {RAMP_ROWS} rows ({RAMP_SECONDS} s)")
print(f"SPC        : {SPC} samples/cycle  at {CYCLE_HZ} Hz")
print(f"Window     : {WINDOW_CYCLES} cycles ({WINDOW_CYCLES * SPC} rows)")
print(f"Threshold  : RMSE > {BAD_THRESHOLD}× median in window")
print(f"Patch      : delete {DELETE_ROWS} rows, duplicate {DUPLICATE_ROWS} rows  "
      f"(net {DUPLICATE_ROWS - DELETE_ROWS:+d} rows per patch)")
print(f"\nSearching for bad cycles...\n")
print(f"  {'Patch':>6}  {'Input row':>10}  {'Time (s)':>10}  {'Net rows':>9}")
print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*9}")

# ── Process ───────────────────────────────────────────────────────────────────
output_chunks = []
patches       = []

# Ramp rows: copy unchanged
output_chunks.append(data[:RAMP_ROWS])
pos = RAMP_ROWS

while pos < N:
    win_end       = min(pos + WINDOW_CYCLES * SPC, N)
    n_full_cycles = (win_end - pos) // SPC

    if n_full_cycles < 3:
        # Not enough cycles for comparison — copy remainder and stop
        output_chunks.append(data[pos:win_end])
        break

    # Segment into (n_full_cycles, SPC, n_ch)
    seg = data[pos : pos + n_full_cycles * SPC].reshape(n_full_cycles, SPC, n_ch)
    det = seg[:, :, det_idx]                         # (n_cycles, SPC, n_det)

    # Median reference cycle
    ref     = np.median(det, axis=0)                 # (SPC, n_det)

    # RMSE of each cycle vs reference
    rmse     = np.sqrt(np.mean((det - ref[np.newaxis]) ** 2, axis=(1, 2)))
    med_rmse = np.median(rmse)

    # Find FIRST cycle above threshold (left-to-right), not just the worst.
    # This ensures an earlier bad cycle is never skipped when a later one
    # has a higher RMSE — after patching the first, the next window will
    # catch any remaining bad cycles.
    bad_idx = next(
        (i for i in range(n_full_cycles)
         if med_rmse > 0 and rmse[i] > BAD_THRESHOLD * med_rmse),
        None
    )

    if bad_idx is not None:
        bad_start = pos + bad_idx * SPC

        # Start patch one cycle before the detected bad cycle
        patch_start = max(pos, bad_start - SPC)

        # Output all good data up to the patch start
        output_chunks.append(data[pos:patch_start])

        # Patch: delete DELETE_ROWS rows, duplicate next DUPLICATE_ROWS rows
        dup_start = patch_start + DELETE_ROWS
        dup_end   = min(dup_start + DUPLICATE_ROWS, N)

        output_chunks.append(data[dup_start:dup_end])   # duplicate copy
        output_chunks.append(data[dup_start:dup_end])   # keep original

        patches.append((patch_start, time_fixed[patch_start]))
        print(f"  {len(patches):>6}  {patch_start:>10,}  "
              f"{time_fixed[patch_start]:>10.4f}  "
              f"{DUPLICATE_ROWS - DELETE_ROWS:>+9d}")

        pos = dup_end   # continue after the duplicated block

    else:
        # No bad cycle in this window — advance by full window
        output_chunks.append(data[pos : pos + n_full_cycles * SPC])
        pos += n_full_cycles * SPC

# ── Assemble ──────────────────────────────────────────────────────────────────
output_data = np.vstack(output_chunks)
M           = len(output_data)
new_time    = np.arange(0, M) / SAMPLE_RATE

print(f"\n{'─'*52}")
if not patches:
    print("  No bad cycles detected.")
print(f"  Patches applied  : {len(patches)}")
print(f"  Input rows       : {N:,}  ({time_fixed[-1]:.4f} s)")
print(f"  Output rows      : {M:,}  ({new_time[-1]:.4f} s)")
print(f"  Net row change   : {M - N:+,}")
print(f"{'─'*52}")

# ── Save-as dialog ────────────────────────────────────────────────────────────
default_name = os.path.splitext(os.path.basename(in_path))[0] + "_aligned.txt"

save_root = tk.Tk()
save_root.withdraw()

out_path = filedialog.asksaveasfilename(
    title="Save aligned output file as...",
    initialdir=os.path.dirname(in_path),
    initialfile=default_name,
    defaultextension=".txt",
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
)

save_root.destroy()

if not out_path:
    print("\nSave cancelled — no file written.")
    raise SystemExit(0)

# ── Write ─────────────────────────────────────────────────────────────────────
print(f"\nWriting: {out_path}")
with open(out_path, "w") as f:
    f.write("time_s\t" + "\t".join(cols) + "\n")
    for i in range(M):
        row_vals = [f"{new_time[i]:.6f}"] + [f"{v:.6f}" for v in output_data[i]]
        f.write("\t".join(row_vals) + "\n")

print(f"Done — {M:,} rows written.")
print(f"  Time : {new_time[0]:.4f} s → {new_time[-1]:.4f} s")
