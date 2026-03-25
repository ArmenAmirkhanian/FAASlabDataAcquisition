# ─── IMPORT LIBRARIES ─────────────────────────────────────────────────────────
import pandas as pd          # For reading Excel files and organizing data
import numpy as np           # For math calculations
import matplotlib.pyplot as plt  # For plotting graphs

# ─── FILE NAMES ───────────────────────────────────────────────────────────────
# Update these to match your actual Excel file names
PYTHON_FILE = r"C:\Users\akumari1\OneDrive - The University of Alabama\FAA Flood Resilience-Civil Const and Env Engineering - Documents\Laboratory Data\FAA_SoilPit\Slab_test\Trial_1kip\Trial_1kip_python.xlsx"        # Your Python recorded Excel file
SIGNAL_FILE = r"C:\Users\akumari1\OneDrive - The University of Alabama\FAA Flood Resilience-Civil Const and Env Engineering - Documents\Laboratory Data\FAA_SoilPit\Slab_test\Trial_1kip\Trial_1kip_signal express.xlsx" # Your Signal Express Excel file

# ─── COMPARISON INTERVAL ──────────────────────────────────────────────────────
COMPARE_EVERY_N = 50    # Compare percentage difference after every 50th data point

# ─── PRESSURE CONVERSION FORMULAS ─────────────────────────────────────────────
# Each function takes raw voltage and returns pressure in kips
def process_soil_plate_pressure(raw):
    # Formula for Soil Plate Pressure sensor (Column N / ai17)
    return -2.86e-5 * raw**2 + 1.0038 * raw + 0.9331

def process_agg_plate_pressure(raw):
    # Formula for Aggregate Plate Pressure sensor (Column O / ai18)
    return -1.34e-4 * raw**2 + 2.5171 * raw - 1.3375

def process_soil_pore_water_pressure(raw):
    # Formula for Soil Pore Water Pressure sensor (Column P / ai19)
    return -2.79e-5 * raw**2 + 1.0006 * raw + 0.4014

def process_agg_pore_water_pressure(raw):
    # Formula for Aggregate Pore Water Pressure sensor (Column Q / ai20)
    return -2.93e-5 * raw**2 + 1.0073 * raw + 0.9260

# ─── LOAD DATA FROM EXCEL ─────────────────────────────────────────────────────
print("Loading Python data...")
# Read Sheet 1 for load data
# header=7 means row 8 is the header row (0-indexed), data starts row 9
py_load_sheet = pd.read_excel(PYTHON_FILE, sheet_name=0, header=7)

# Read Sheet 2 for displacement, voltage, strain
# header=0 means first row is the header row
py_data_sheet = pd.read_excel(PYTHON_FILE, sheet_name=1, header=0)

print("Loading Signal Express data...")
se_load_sheet = pd.read_excel(SIGNAL_FILE, sheet_name=0, header=7)
se_data_sheet = pd.read_excel(SIGNAL_FILE, sheet_name=1, header=0)

# ─── EXTRACT LOAD (Sheet 1, Column C) ─────────────────────────────────────────
# Column C is index 2 (A=0, B=1, C=2)
py_load_raw = py_load_sheet.iloc[:, 2].dropna().values.astype(float)
se_load_raw = se_load_sheet.iloc[:, 2].dropna().values.astype(float)
# iloc[:, 2] means: all rows, column index 2 (Column C)
# .dropna() removes any empty cells
# .values converts to a simple number array
# .astype(float) makes sure all values are numbers

# ─── TARE LOAD (subtract first value so it starts at 0) ───────────────────────
py_load = py_load_raw - py_load_raw[0]
se_load = se_load_raw - se_load_raw[0]
# Taring means subtracting the very first reading from all readings
# This sets the starting point to zero

# ─── EXTRACT DISPLACEMENT (Sheet 2, Columns B to M = index 1 to 12) ───────────
# Python file displacement column names
py_disp_cols = [
    'DCDT_Right_Slab_A1', 'DCDT_Right_Slab_A2', 'DCDT_Right_Slab_A3',
    'DCDT_Right_Slab_B1', 'DCDT_Right_Slab_B3', 'DCDT_Left_Slab_B1',
    'DCDT_Left_Slab_B2_Bot', 'DCDT_Left_Slab_B3', 'DCDT_Left_Slab_C1',
    'DCDT_Left_Slab_C2', 'DCDT_Left_Slab_C3', 'DCDT_Beam_B2_Top'
]

# Signal Express displacement column names
se_disp_cols = [
    'Subset Voltage - ai0_DCDT_Right_Slab_A1',
    'Subset Voltage - ai1_DCDT_Right_Slab_A2',
    'Subset Voltage - ai2_DCDT_Right_Slab_A3',
    'Subset Voltage - ai3_DCDT_Right_Slab_B1',
    'Subset Voltage - ai4_DCDT_Right_Slab_B3',
    'Subset Voltage - ai5_DCDT_Left_Slab_B1',
    'Subset Voltage - ai6_DCDT_Left_Slab_B2_Bot',
    'Subset Voltage - ai7_DCDT_Left_Slab_B3',
    'Subset Voltage - ai8_DCDT_Left_Slab_C1',
    'Subset Voltage - ai9_DCDT_Left_Slab_C2',
    'Subset Voltage - ai10_DCDT_Left_Slab_C3',
    'Subset Voltage - ai11_DCDT_Beam_B2_Top'
]

# Extract displacement data and tare (subtract first row from all rows)
py_disp = py_data_sheet[py_disp_cols].values.astype(float)
se_disp = se_data_sheet[se_disp_cols].values.astype(float)
py_disp = py_disp - py_disp[0, :]   # Subtract first row from every row
se_disp = se_disp - se_disp[0, :]   # [0, :] means first row, all columns

# ─── EXTRACT STRAIN (Sheet 2, Columns R to Y) ─────────────────────────────────
py_strain_cols = [
    'SG_2E_top', 'SG_3E_top', 'SG_4E_top', 'SG_4E_bot',
    'SG_5E_top', 'SG_5E_bot', 'SG_6E_top', 'SG_7E_top'
]

se_strain_cols = [
    "Subset Strain - ai0_SG_2'E_top", "Subset Strain - ai1_SG_3'E_top",
    "Subset Strain - ai2_SG_4'E_top", "Subset Strain - ai3_SG_4'E_bot",
    "Subset Strain - ai4_SG_5'E_top", "Subset Strain - ai5_SG_5'E_bot",
    "Subset Strain - ai6_SG_6'E_top", "Subset Strain - ai7_SG_7'E_top"
]

# Extract strain data and tare
py_strain = py_data_sheet[py_strain_cols].values.astype(float)
se_strain = se_data_sheet[se_strain_cols].values.astype(float)
py_strain = py_strain - py_strain[0, :]
se_strain = se_strain - se_strain[0, :]

# ─── EXTRACT VOLTAGE AND CONVERT TO PRESSURE ──────────────────────────────────
# Python voltage column names
py_volt_cols = ['volt_ch17', 'volt_ch18', 'volt_ch19', 'volt_ch20']

# Signal Express voltage column names
se_volt_cols = [
    'Subset Voltage - ai17_Soil_Plate_Pressure',
    'Subset Voltage - ai18_Agg_Plate_Pressure',
    'Subset Voltage - ai19_Soil_Pore_Water_Pressure',
    'Subset Voltage - ai20_Agg_Pore_Water_Pressure'
]

# Extract raw voltage values
py_volt_raw = py_data_sheet[py_volt_cols].values.astype(float)
se_volt_raw = se_data_sheet[se_volt_cols].values.astype(float)

# Apply pressure formulas to each voltage channel
# np.vectorize allows the formula to work on entire arrays at once
py_pressure = np.column_stack([
    np.vectorize(process_soil_plate_pressure)(py_volt_raw[:, 0]),
    np.vectorize(process_agg_plate_pressure)(py_volt_raw[:, 1]),
    np.vectorize(process_soil_pore_water_pressure)(py_volt_raw[:, 2]),
    np.vectorize(process_agg_pore_water_pressure)(py_volt_raw[:, 3])
])
# np.column_stack combines the 4 processed columns back into one table
# [:, 0] means all rows, column 0 (first voltage channel)

se_pressure = np.column_stack([
    np.vectorize(process_soil_plate_pressure)(se_volt_raw[:, 0]),
    np.vectorize(process_agg_plate_pressure)(se_volt_raw[:, 1]),
    np.vectorize(process_soil_pore_water_pressure)(se_volt_raw[:, 2]),
    np.vectorize(process_agg_pore_water_pressure)(se_volt_raw[:, 3])
])

# Tare pressure (subtract first reading)
py_pressure = py_pressure - py_pressure[0, :]
se_pressure = se_pressure - se_pressure[0, :]

# ─── PERCENTAGE DIFFERENCE FUNCTION ───────────────────────────────────────────
def percent_diff(val1, val2):
    # Calculates percentage difference between two values
    # Uses average of both as the reference (standard engineering method)
    avg = (np.abs(val1) + np.abs(val2)) / 2
    # np.where avoids dividing by zero — if avg is 0, result is 0
    return np.where(avg != 0, np.abs(val1 - val2) / avg * 100, 0)

# ─── PRINT PERCENTAGE DIFFERENCE TABLE ────────────────────────────────────────
def print_comparison_table(py_vals, se_vals, label, col_names):
    print(f"\n{'='*60}")
    print(f"PERCENTAGE DIFFERENCE — {label} (every {COMPARE_EVERY_N} points)")
    print(f"{'='*60}")

    # Loop through data every 50th point
    indices = range(0, min(len(py_vals), len(se_vals)), COMPARE_EVERY_N)
    for i in indices:
        diffs = percent_diff(py_vals[i], se_vals[i])
        # If diffs is a single number (1D), wrap it in a list
        if np.ndim(diffs) == 0:
            diffs = [diffs]
        diff_str = "  ".join([f"{n}: {d:.2f}%" for n, d in zip(col_names, diffs)])
        print(f"  Point {i+1:>5}: {diff_str}")

# Print tables for all parameters
print_comparison_table(py_load, se_load, "LOAD (kips)", ["Load"])
print_comparison_table(py_disp, se_disp, "DISPLACEMENT", py_disp_cols)
print_comparison_table(py_strain, se_strain, "STRAIN", py_strain_cols)
pressure_names = ["Soil Plate", "Agg Plate", "Soil Pore Water", "Agg Pore Water"]
print_comparison_table(py_pressure, se_pressure, "PRESSURE", pressure_names)

# ─── PLOTTING ─────────────────────────────────────────────────────────────────
# Use the shorter dataset length to avoid index errors
min_load_len   = min(len(py_load), len(se_load))
min_data_len   = min(len(py_disp), len(se_disp))

# Short names for displacement legend (easier to read on plot)
disp_short = ['A1R', 'A2R', 'A3R', 'B1R', 'B3R', 'B1L', 'B2L', 'B3L', 'C1L', 'C2L', 'C3L', 'Beam']
strain_short = ['SG2E_t', 'SG3E_t', 'SG4E_t', 'SG4E_b', 'SG5E_t', 'SG5E_b', 'SG6E_t', 'SG7E_t']

# ── Plot 1: Load vs Displacement ──────────────────────────────────────────────
fig1, axes1 = plt.subplots(4, 3, figsize=(18, 16))
# Creates a grid of 4 rows x 3 columns = 12 subplots, one per displacement channel
axes1 = axes1.flatten()
# .flatten() converts the 2D grid into a simple list so we can loop through it

for i in range(12):   # Loop through all 12 displacement channels
    ax = axes1[i]
    ax.plot(py_disp[:min_data_len, i], py_load[:min_data_len],
            label='Python', color='blue', linewidth=1)
    ax.plot(se_disp[:min_data_len, i], se_load[:min_data_len],
            label='Signal Express', color='red', linewidth=1, alpha=0.7)
    ax.set_title(disp_short[i], fontsize=10)
    ax.set_xlabel('Displacement')
    ax.set_ylabel('Load (kips)')
    ax.legend(fontsize=7)
    ax.grid(True)

fig1.suptitle('Load vs Displacement — Python vs Signal Express', fontsize=14)
# suptitle adds a big title at the top of the entire figure
plt.tight_layout()
plt.savefig('load_vs_displacement.png', dpi=150)
# Saves the plot as an image file in your project folder
print("\nSaved: load_vs_displacement.png")

# ── Plot 2: Load vs Strain ─────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 10))
# Creates a grid of 2 rows x 4 columns = 8 subplots, one per strain channel
axes2 = axes2.flatten()

for i in range(8):    # Loop through all 8 strain channels
    ax = axes2[i]
    ax.plot(py_strain[:min_data_len, i], py_load[:min_data_len],
            label='Python', color='blue', linewidth=1)
    ax.plot(se_strain[:min_data_len, i], se_load[:min_data_len],
            label='Signal Express', color='red', linewidth=1, alpha=0.7)
    ax.set_title(strain_short[i], fontsize=10)
    ax.set_xlabel('Strain')
    ax.set_ylabel('Load (kips)')
    ax.legend(fontsize=7)
    ax.grid(True)

fig2.suptitle('Load vs Strain — Python vs Signal Express', fontsize=14)
plt.tight_layout()
plt.savefig('load_vs_strain.png', dpi=150)
print("Saved: load_vs_strain.png")

# ── Plot 3: Load vs Pressure ───────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
# Creates a 2x2 grid = 4 subplots, one per pressure channel
axes3 = axes3.flatten()

for i in range(4):    # Loop through all 4 pressure channels
    ax = axes3[i]
    ax.plot(py_pressure[:min_data_len, i], py_load[:min_data_len],
            label='Python', color='blue', linewidth=1)
    ax.plot(se_pressure[:min_data_len, i], se_load[:min_data_len],
            label='Signal Express', color='red', linewidth=1, alpha=0.7)
    ax.set_title(pressure_names[i], fontsize=10)
    ax.set_xlabel('Pressure')
    ax.set_ylabel('Load (kips)')
    ax.legend(fontsize=7)
    ax.grid(True)

fig3.suptitle('Load vs Pressure — Python vs Signal Express', fontsize=14)
plt.tight_layout()
plt.savefig('load_vs_pressure.png', dpi=150)
print("Saved: load_vs_pressure.png")

plt.show()
# Opens all three plot windows on your screen
print("\nDone! All plots saved to your project folder.")