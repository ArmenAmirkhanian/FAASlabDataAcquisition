# ─── OFFLINE RAW-FILE POST-PROCESSOR ──────────────────────────
# Reconstructs the processed (tared) and calibrated (converted, untared) files
# from a data_raw_*.txt file produced by DataProcessing.py, using the same
# conversion formulas and taring logic that used to run live during acquisition.
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd

# ─── CONFIGURATION (must match DataProcessing.py) ──────────────
RAMP_SAMPLES = 160        # Rows used to lock the tare/baseline (10s ramp/dwell @ 16Hz)
KPA_TO_PSI   = 0.145038   # kPa → psi conversion

# Per-channel V → inches scale (0 V = 0 in, 10 V = full stroke)
DISP_SCALE = [
    3.937 / 10.0,   # ai0  DCDT_Right_Slab_A1
    3.937 / 10.0,   # ai1  DCDT_Right_Slab_A2
    3.937 / 10.0,   # ai2  DCDT_Right_Slab_A3
    3.937 / 10.0,   # ai3  DCDT_Right_Slab_B1
    1.969 / 10.0,   # ai4  DCDT_Right_Slab_B3
    3.937 / 10.0,   # ai5  DCDT_Left_Slab_B1
    3.937 / 10.0,   # ai6  DCDT_Left_Slab_B2_Bot
    1.969 / 10.0,   # ai7  DCDT_Left_Slab_B3
    3.937 / 10.0,   # ai8  DCDT_Left_Slab_C1
    3.937 / 10.0,   # ai9  DCDT_Left_Slab_C2
    1.969 / 10.0,   # ai10 DCDT_Left_Slab_C3
    0.9843 / 10.0,  # ai11 DCDT_Beam_B2_Top
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
    "SG_1E_top", "SG_2E_top", "SG_3E_top", "SG_3E_bot", "SG_4E_top", "SG_4E_bot",
    "SG_5E_top", "SG_5E_bot", "SG_6E_top", "SG_7E_top"
]
disp_names = [
    "DCDT_Right_Slab_A1", "DCDT_Right_Slab_A2", "DCDT_Right_Slab_A3",
    "DCDT_Right_Slab_B1", "DCDT_Right_Slab_B3",
    "DCDT_Left_Slab_B1",  "DCDT_Left_Slab_B2_Bot", "DCDT_Left_Slab_B3",
    "DCDT_Left_Slab_C1",  "DCDT_Left_Slab_C2", "DCDT_Left_Slab_C3",
    "DCDT_Beam_B2_Top"
]
press_volt_cols = [
    "Soil_plate_volt_ch17", "Agg_plate_volt_ch18",
    "Soil_pore_water_volt_ch19", "Agg_pore_water_volt_ch20"
]
press_names_kpa = [
    "soil_plate_pressure_kPa", "agg_plate_pressure_kPa",
    "soil_pore_water_pressure_kPa", "agg_pore_water_pressure_kPa"
]
press_names_psi = [
    "soil_plate_pressure_psi", "agg_plate_pressure_psi",
    "soil_pore_water_pressure_psi", "agg_pore_water_pressure_psi"
]


def process_raw_file(raw_path):
    """Read a data_raw_*.txt file and return (processed_df, calibrated_df)."""
    df = pd.read_csv(raw_path, sep="\t")

    if len(df) <= RAMP_SAMPLES:
        raise ValueError(
            f"File only has {len(df)} rows — need more than {RAMP_SAMPLES} "
            f"(the ramp/tare window) to compute a baseline."
        )

    # ── Convert raw → physical units (same formulas as DataProcessing.py) ──
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


def output_paths_for(raw_path):
    """Derive processed/calibrated filenames from the raw file's own timestamp suffix."""
    out_dir = os.path.dirname(raw_path)
    base = os.path.basename(raw_path)
    if base.startswith("data_raw_") and base.endswith(".txt"):
        suffix = base[len("data_raw_"):-len(".txt")]
    else:
        suffix = os.path.splitext(base)[0]
    proc_filename = os.path.join(out_dir, f"data_processed_{suffix}.txt")
    cal_filename  = os.path.join(out_dir, f"data_calibrated_{suffix}.txt")
    return proc_filename, cal_filename


def main():
    root = tk.Tk()
    root.withdraw()

    raw_path = filedialog.askopenfilename(
        title="Select a raw DAQ file to process",
        filetypes=[("Raw data files", "data_raw_*.txt"), ("Text files", "*.txt"), ("All files", "*.*")]
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

    proc_filename, cal_filename = output_paths_for(raw_path)
    proc_df.to_csv(proc_filename, sep="\t", index=False, float_format="%.6f")
    cal_df.to_csv(cal_filename,  sep="\t", index=False, float_format="%.6f")

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
