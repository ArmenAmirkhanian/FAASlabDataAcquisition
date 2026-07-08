"""
postprocess_edit.py — Periodic row deletion and duplication.

For each cycle starting at START_S and repeating every PERIOD_S in the input:
  1. Delete  DELETE_ROWS rows
  2. Insert  DUPLICATE_ROWS rows (copy of rows immediately after deletion)
  3. Keep original DUPLICATE_ROWS rows  (original data stays)
  4. Keep remaining rows to end of period  (= PERIOD_ROWS - DELETE - DUPLICATE)

Net change per cycle: DUPLICATE_ROWS - DELETE_ROWS rows.

Input time is fixed to start at 1/SAMPLE_RATE (0.0625 s at 16 Hz).
Output time is rebuilt from 1/SAMPLE_RATE at SAMPLE_RATE Hz.

Usage:
    python postprocess_edit.py
"""

import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16       # Hz
START_S        = 6.0      # time in input where first DELETE begins (seconds)
PERIOD_S       = 10.0     # spacing between cycle starts in input (seconds)
DELETE_ROWS    = 40       # rows removed at start of each cycle
DUPLICATE_ROWS = 48       # rows copied and inserted right after deletion

# Derived — do not edit
PERIOD_ROWS = int(PERIOD_S * SAMPLE_RATE)
KEEP_ROWS   = PERIOD_ROWS - DELETE_ROWS - DUPLICATE_ROWS

# ── Validate config ───────────────────────────────────────────────────────────
if KEEP_ROWS < 0:
    raise ValueError(
        f"DELETE_ROWS ({DELETE_ROWS}) + DUPLICATE_ROWS ({DUPLICATE_ROWS}) "
        f"= {DELETE_ROWS + DUPLICATE_ROWS} exceeds PERIOD_ROWS ({PERIOD_ROWS}). "
        f"Reduce DELETE or DUPLICATE, or increase PERIOD_S."
    )

# ── File browser ──────────────────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()

in_path = filedialog.askopenfilename(
    title="Select input file (reference — will not be modified)",
    initialdir=os.path.dirname(os.path.abspath(__file__)),
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
)

if not in_path:
    messagebox.showinfo("Cancelled", "No file selected. Exiting.")
    root.destroy()
    raise SystemExit(0)

root.destroy()

# ── Load ──────────────────────────────────────────────────────────────────────
df   = pd.read_csv(in_path, sep="\t")
cols = [c for c in df.columns if c != "time_s"]
data = df[cols].values.astype(float)
N    = len(data)

# Fix input time: row i (0-indexed) → t = (i+1) / SAMPLE_RATE
# Row 1 → 0.0625 s,  Row 96 → 6.0 s,  etc.
time_fixed = np.arange(1, N + 1) / SAMPLE_RATE

print(f"\nInput    : {os.path.basename(in_path)}")
print(f"Rows     : {N:,}")
print(f"Duration : {time_fixed[-1]:.4f} s  ({time_fixed[-1]/60:.2f} min)")
print(f"\n{'─'*52}")
print(f"  START_S        = {START_S} s")
print(f"  PERIOD_S       = {PERIOD_S} s  ({PERIOD_ROWS} rows per cycle)")
print(f"  DELETE_ROWS    = {DELETE_ROWS}")
print(f"  DUPLICATE_ROWS = {DUPLICATE_ROWS}")
print(f"  KEEP_ROWS      = {KEEP_ROWS}  "
      f"(= {PERIOD_ROWS} − {DELETE_ROWS} − {DUPLICATE_ROWS})")
print(f"  Net gain       = +{DUPLICATE_ROWS - DELETE_ROWS} rows per cycle")
print(f"{'─'*52}")

# ── Find cycle start rows ─────────────────────────────────────────────────────
first_row = int(np.searchsorted(time_fixed, START_S))

cycle_starts = []
r = first_row
while r < N:
    cycle_starts.append(r)
    r += PERIOD_ROWS

n_cycles = len(cycle_starts)
print(f"\nCycles identified: {n_cycles}")
print(f"  {'#':>4}   {'Input row':>10}   {'Input time (s)':>14}   {'Output rows':>12}")

out_row_counter = first_row  # rows before first cycle are copied unchanged
for k, cr in enumerate(cycle_starts):
    dup_start = cr + DELETE_ROWS
    dup_end   = min(dup_start + DUPLICATE_ROWS, N)
    keep_end  = min(cr + PERIOD_ROWS, N)
    cycle_out = (dup_end - dup_start) * 2 + (keep_end - dup_end)
    print(f"  {k+1:>4}   {cr:>10,}   {time_fixed[cr]:>14.4f}   {cycle_out:>12,}")
    out_row_counter += cycle_out

# ── Build output ──────────────────────────────────────────────────────────────
chunks  = []
prev_end = 0

for r in cycle_starts:
    # Copy input rows before this cycle start (unchanged)
    if r > prev_end:
        chunks.append(data[prev_end:r])

    dup_start = r + DELETE_ROWS
    dup_end   = min(dup_start + DUPLICATE_ROWS, N)
    keep_end  = min(r + PERIOD_ROWS, N)

    # Delete: rows r → r+DELETE_ROWS-1 are skipped (not added to output)

    # Duplicate: insert copy of the block right after deletion
    if dup_start < N:
        chunks.append(data[dup_start:dup_end])

    # Keep original: same block stays in output
    if dup_start < N:
        chunks.append(data[dup_start:dup_end])

    # Keep rest: remaining rows in this period
    if dup_end < keep_end:
        chunks.append(data[dup_end:keep_end])

    prev_end = keep_end

# Copy any rows after the last cycle
if prev_end < N:
    chunks.append(data[prev_end:])

# ── Assemble ──────────────────────────────────────────────────────────────────
output_data = np.vstack(chunks)
M           = len(output_data)
new_time    = np.arange(1, M + 1) / SAMPLE_RATE

expected_gain = n_cycles * (DUPLICATE_ROWS - DELETE_ROWS)
print(f"\n{'─'*52}")
print(f"  Input rows     : {N:,}")
print(f"  Output rows    : {M:,}")
print(f"  Actual gain    : {M - N:+,} rows")
print(f"  Expected gain  : {expected_gain:+,} rows  ({n_cycles} cycles × +{DUPLICATE_ROWS - DELETE_ROWS})")
print(f"  Output duration: {new_time[-1]:.4f} s  ({new_time[-1]/60:.2f} min)")
print(f"{'─'*52}")

# ── Save-as dialog ────────────────────────────────────────────────────────────
default_name = os.path.splitext(os.path.basename(in_path))[0] + "_edited.txt"

save_root = tk.Tk()
save_root.withdraw()

out_path = filedialog.asksaveasfilename(
    title="Save edited output file as...",
    initialdir=os.path.dirname(in_path),
    initialfile=default_name,
    defaultextension=".txt",
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
)

save_root.destroy()

if not out_path:
    print("\nSave cancelled — no file written.")
    raise SystemExit(0)

# ── Write output ──────────────────────────────────────────────────────────────
print(f"\nWriting: {out_path}")
with open(out_path, "w") as f:
    f.write("time_s\t" + "\t".join(cols) + "\n")
    for i in range(M):
        row_vals = [f"{new_time[i]:.6f}"] + [f"{v:.6f}" for v in output_data[i]]
        f.write("\t".join(row_vals) + "\n")

print(f"Done — {M:,} rows written.")
print(f"  Time range : {new_time[0]:.4f} s → {new_time[-1]:.4f} s")
