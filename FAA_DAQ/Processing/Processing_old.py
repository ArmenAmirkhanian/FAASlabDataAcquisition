# ─── OFFLINE RAW-FILE POST-PROCESSOR (OLD 8-STRAIN-CHANNEL FORMAT) ─────
# Same as Processing.py, but for the older data set that has:
#   - only 8 strain gauges (no SG_1E_top, no SG_3E_bot)
#   - no DCDT_Left_Slab_B2_Bot channel (11 DCDT channels instead of 12)
#   - raw pressure columns named volt_ch17..volt_ch20 (not Soil_plate_volt_ch17 etc.)
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd

# ─── CONFIGURATION ──────────────────────────────────────────────
RAMP_SAMPLES = 80        # Rows used to lock the tare/baseline (10s ramp/dwell @ 16Hz)
KPA_TO_PSI   = 0.145038   # kPa → psi conversion

# Per-channel V → inches scale (0 V = 0 in, 10 V = full stroke)
# TODO: confirm these against the old DataProcessing/config file — currently copied
# from Processing.py with DCDT_Left_Slab_B2_Bot removed.
DISP_SCALE = [
    3.937 / 10.0,   # DCDT_Right_Slab_A1
    3.937 / 10.0,   # DCDT_Right_Slab_A2
    3.937 / 10.0,   # DCDT_Right_Slab_A3
    3.937 / 10.0,   # DCDT_Right_Slab_B1
    1.969 / 10.0,   # DCDT_Right_Slab_B3
    3.937 / 10.0,   # DCDT_Left_Slab_B1
    1.969 / 10.0,   # DCDT_Left_Slab_B3
    3.937 / 10.0,   # DCDT_Left_Slab_C1
    3.937 / 10.0,   # DCDT_Left_Slab_C2
    3.937 / 10.0,   # DCDT_Left_Slab_C3
    0.9843 / 10.0,  # DCDT_Beam_B2_Top
]

# ─── PRESSURE CONVERSION FORMULAS (one per channel) ──────────
def process_soil_plate_pressure(raw):
    return -2.86e-5 * raw**2 + 1.0038 * raw + 0.9331

def process_agg_plate_pressure(raw):
    return -1.34e-4 * raw**2 + 2.5171 * raw - 1.3375

def process_soil_pore_water_pressure(raw):
    return -2.79e-5 * raw**2 + 1.0006 * raw + 0.4014

def process_agg_pore_water_pressure(raw):
    return -2.93e-5 * raw**2 + 1.0073 * raw + 0.9260

pressure_formulas = [
    process_soil_plate_pressure,
    process_agg_plate_pressure,
    process_soil_pore_water_pressure,
    process_agg_pore_water_pressure,
]

strain_names = [
    "SG_2E_top", "SG_3E_top", "SG_4E_top", "SG_4E_bot",
    "SG_5E_top", "SG_5E_bot", "SG_6E_top", "SG_7E_top"
]
disp_names = [
    "DCDT_Right_Slab_A1", "DCDT_Right_Slab_A2", "DCDT_Right_Slab_A3",
    "DCDT_Right_Slab_B1", "DCDT_Right_Slab_B3",
    "DCDT_Left_Slab_B1",  "DCDT_Left_Slab_B3",
    "DCDT_Left_Slab_C1",  "DCDT_Left_Slab_C2", "DCDT_Left_Slab_C3",
    "DCDT_Beam_B2_Top"
]
press_volt_cols = ["volt_ch17", "volt_ch18", "volt_ch19", "volt_ch20"]
press_names_kpa = [
    "soil_plate_pressure_kPa", "agg_plate_pressure_kPa",
    "soil_pore_water_pressure_kPa", "agg_pore_water_pressure_kPa"
]
press_names_psi = [
    "soil_plate_pressure_psi", "agg_plate_pressure_psi",
    "soil_pore_water_pressure_psi", "agg_pore_water_pressure_psi"
]


def process_raw_file(raw_path):
    """Read an old-format DAQ file and return (processed_df, calibrated_df)."""
    df = pd.read_csv(raw_path, sep="\t")

    if len(df) <= RAMP_SAMPLES:
        raise ValueError(
            f"File only has {len(df)} rows — need more than {RAMP_SAMPLES} "
            f"(the ramp/tare window) to compute a baseline."
        )

    # ── Convert raw → physical units ──
    disp_in = df[disp_names].to_numpy() * DISP_SCALE
    press_kpa = np.column_stack([
        formula(df[col].to_numpy() * 1000)  # formulas expect millivolts
        for formula, col in zip(pressure_formulas, press_volt_cols)
    ])
    strain = df[strain_names].to_numpy()

    # ── Tare: average of the first RAMP_SAMPLES rows (the ramp/dwell window) ──
    baseline_disp   = disp_in[:RAMP_SAMPLES].mean(axis=0)
    baseline_press  = press_kpa[:RAMP_SAMPLES].mean(axis=0)
    baseline_strain = strain[:RAMP_SAMPLES].mean(axis=0)

    disp_tared      = disp_in - baseline_disp
    press_tared_kpa = press_kpa - baseline_press
    press_tared_psi = press_tared_kpa * KPA_TO_PSI
    strain_tared    = strain - baseline_strain

    t = df["time_s"].to_numpy()

    cal_df = pd.DataFrame(disp_in, columns=disp_names)
    cal_df.insert(0, "time_s", t)
    for name, col in zip(press_names_kpa, press_kpa.T):
        cal_df[name] = col

    proc_df = pd.DataFrame(disp_tared, columns=disp_names)
    proc_df.insert(0, "time_s", t)
    for name, col in zip(press_names_psi, press_tared_psi.T):
        proc_df[name] = col
    for name, col in zip(strain_names, strain_tared.T):
        proc_df[name] = col

    return proc_df, cal_df


def win_long_path(path):
    """Prefix with \\\\?\\ so Windows accepts paths over the 260-char MAX_PATH limit."""
    if sys.platform == "win32":
        abs_path = os.path.abspath(path)
        if not abs_path.startswith("\\\\?\\"):
            abs_path = "\\\\?\\" + abs_path
        return abs_path
    return path


def output_paths_for(out_dir, set_label):
    """Build processed/calibrated filenames from a short set label (e.g. 'Set_1')."""
    proc_filename = os.path.join(out_dir, f"{set_label}_processed.txt")
    cal_filename  = os.path.join(out_dir, f"{set_label}_calibrated.txt")
    return proc_filename, cal_filename


def main():
    root = tk.Tk()
    root.withdraw()

    raw_path = filedialog.askopenfilename(
        title="Select an old-format DAQ file to process",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not raw_path:
        print("No file selected. Exiting.")
        root.destroy()
        return

    try:
        proc_df, cal_df = process_raw_file(raw_path)
    except Exception as e:
        messagebox.showerror("Processing Failed", str(e))
        root.destroy()
        raise

    out_dir = filedialog.askdirectory(title="Select a folder to save the processed/calibrated files")
    if not out_dir:
        print("No output folder selected. Exiting.")
        root.destroy()
        return

    set_label = simpledialog.askstring(
        "Set Number",
        "Enter the set label (e.g. Set_1):",
        parent=root
    )
    if not set_label:
        print("No set label entered. Exiting.")
        root.destroy()
        return

    proc_filename, cal_filename = output_paths_for(out_dir, set_label)
    proc_df.to_csv(win_long_path(proc_filename), sep="\t", index=False, float_format="%.6f")
    cal_df.to_csv(win_long_path(cal_filename),  sep="\t", index=False, float_format="%.6f")

    print(f"Processed data saved to  {proc_filename}")
    print(f"Calibrated data saved to {cal_filename}")
    messagebox.showinfo(
        "Processing Complete",
        f"Processed and calibrated files saved:\n\n  • {proc_filename}\n  • {cal_filename}"
    )
    root.destroy()


# ─── RUN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
