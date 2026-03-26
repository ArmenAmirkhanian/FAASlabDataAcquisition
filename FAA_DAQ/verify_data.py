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

# ─── LOAD ALIGNMENT SHIFT ─────────────────────────────────────────────────────
# Shifts the entire Python dataset to align its load curve with Signal Express.
# None  → auto-detect using cross-correlation (recommended first run)
# +N    → skip first N rows from Python start  (Python started recording N rows too early)
# -N    → skip first N rows from SE start      (SE started recording N rows too early)
PY_ROW_OFFSET = None

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

# ─── EXTRACT TIME (Sheet 1, Column A, data starts row 9) ──────────────────────
# Column A is index 0; header=7 already skips rows 1-8, so iloc gives rows from row 9
py_time_raw = py_load_sheet.iloc[:, 0].dropna().values
se_time_raw = se_load_sheet.iloc[:, 0].dropna().values

# ─── TARE LOAD (subtract first value so it starts at 0) ───────────────────────
py_load = py_load_raw - py_load_raw[0]
se_load = se_load_raw - se_load_raw[0]

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

# ─── STEP 1: TRIM EACH DATASET INTERNALLY ────────────────────────────────────
# Make sure load/time and sensor arrays are the same length within each file
py_n = min(len(py_load), len(py_disp), len(py_time_raw))
se_n = min(len(se_load), len(se_disp), len(se_time_raw))

py_load     = py_load[:py_n];      se_load     = se_load[:se_n]
py_time     = py_time_raw[:py_n];  se_time     = se_time_raw[:se_n]
py_disp     = py_disp[:py_n];      se_disp     = se_disp[:se_n]
py_strain   = py_strain[:py_n];    se_strain   = se_strain[:se_n]
py_pressure = py_pressure[:py_n];  se_pressure = se_pressure[:se_n]

# ─── STEP 2: SHIFT PYTHON TO ALIGN LOAD CURVES ───────────────────────────────
# Auto-detect the row offset using cross-correlation on the load signals.
# Both load arrays are resampled to the same length so their shapes don't matter.
# The correlation peak tells us how many rows one dataset is ahead of the other.
def _detect_offset(py_ld, se_ld):
    n = max(len(py_ld), len(se_ld))
    # Resample both to length n using linear interpolation
    py_r = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(py_ld)), py_ld)
    se_r = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(se_ld)), se_ld)
    # Cross-correlate: find how much py_r must shift to match se_r
    corr = np.correlate(se_r - se_r.mean(), py_r - py_r.mean(), mode='full')
    lag = int(np.argmax(corr)) - (n - 1)
    # lag > 0 → SE started earlier; convert to SE rows and return as negative offset
    # lag < 0 → Python started earlier; convert to Python rows and return as positive offset
    if lag > 0:
        return -int(round(lag * len(se_ld) / n))   # trim SE start
    else:
        return  int(round(-lag * len(py_ld) / n))  # trim Python start

offset = PY_ROW_OFFSET if PY_ROW_OFFSET is not None else _detect_offset(py_load, se_load)

if offset > 0:
    # Python started recording earlier — skip its first `offset` rows
    py_load     = py_load[offset:];     se_load     = se_load
    py_time     = py_time[offset:];     se_time     = se_time
    py_disp     = py_disp[offset:];     se_disp     = se_disp
    py_strain   = py_strain[offset:];   se_strain   = se_strain
    py_pressure = py_pressure[offset:]; se_pressure = se_pressure
    print(f"Shift applied: skipped first {offset} Python rows (Python started earlier)")
elif offset < 0:
    # SE started recording earlier — skip its first `|offset|` rows
    n = -offset
    py_load     = py_load;     se_load     = se_load[n:]
    py_time     = py_time;     se_time     = se_time[n:]
    py_disp     = py_disp;     se_disp     = se_disp[n:]
    py_strain   = py_strain;   se_strain   = se_strain[n:]
    py_pressure = py_pressure; se_pressure = se_pressure[n:]
    print(f"Shift applied: skipped first {n} SE rows (SE started earlier)")
else:
    print("No shift applied — load curves already aligned")

# ─── STEP 3: MATCH BY LOAD VALUE ─────────────────────────────────────────────
# After the whole-dataset shift above the two curves start at the same load state.
# Because sample rates still differ, rows still don't correspond 1-to-1.
# For each Python row, find the SE row with the closest load value (nearest neighbour).
se_match_idx = np.array([np.argmin(np.abs(se_load - lv)) for lv in py_load])

# Build matched SE arrays (same length as Python, compared at equal load levels)
se_disp_m   = se_disp[se_match_idx]
se_strain_m = se_strain[se_match_idx]
se_press_m  = se_pressure[se_match_idx]

# ─── PERCENTAGE DIFFERENCE FUNCTION ───────────────────────────────────────────
def percent_diff(val1, val2):
    avg = (np.abs(val1) + np.abs(val2)) / 2
    return np.where(avg != 0, np.abs(val1 - val2) / avg * 100, 0)

# ─── PRINT PERCENTAGE DIFFERENCE TABLE ────────────────────────────────────────
def print_comparison_table(py_vals, se_vals_matched, label, col_names):
    # Compares Python rows vs the SE rows matched to the same load value
    print(f"\n{'='*60}")
    print(f"PERCENTAGE DIFFERENCE — {label} (every {COMPARE_EVERY_N} points)")
    print(f"{'='*60}")
    indices = range(0, len(py_vals), COMPARE_EVERY_N)
    for i in indices:
        diffs = percent_diff(py_vals[i], se_vals_matched[i])
        if np.ndim(diffs) == 0:
            diffs = [diffs]
        diff_str = "  ".join([f"{n}: {d:.2f}%" for n, d in zip(col_names, diffs)])
        print(f"  Load {py_load[i]:.4f} kips (PY row {i+1}): {diff_str}")

pressure_names = ["Soil Plate", "Agg Plate", "Soil Pore Water", "Agg Pore Water"]

print_comparison_table(py_disp,     se_disp_m,   "DISPLACEMENT", py_disp_cols)
print_comparison_table(py_strain,   se_strain_m, "STRAIN",       py_strain_cols)
print_comparison_table(py_pressure, se_press_m,  "PRESSURE",     pressure_names)

# ─── COMBINED 2×2 OVERVIEW FIGURE ─────────────────────────────────────────────
disp_short   = ['A1R', 'A2R', 'A3R', 'B1R', 'B3R', 'B1L', 'B2L', 'B3L', 'C1L', 'C2L', 'C3L', 'Beam']
strain_short = ['SG2E_t', 'SG3E_t', 'SG4E_t', 'SG4E_b', 'SG5E_t', 'SG5E_b', 'SG6E_t', 'SG7E_t']
press_short  = ['SoilPlate', 'AggPlate', 'SoilPore', 'AggPore']

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Python vs Signal Express — Load-Matched Comparison', fontsize=14)

# ── Panel 1: Load vs Displacement (all 12 channels overlaid) ──────────────────
for i in range(12):
    ax1.plot(py_disp[:, i],   py_load, color='blue', linewidth=0.8,
             label=f'PY {disp_short[i]}' if i == 0 else '_')
    ax1.plot(se_disp_m[:, i], py_load, color='red',  linewidth=0.8, alpha=0.7,
             label=f'SE {disp_short[i]}' if i == 0 else '_')
ax1.set_title('Load vs Displacement')
ax1.set_xlabel('Displacement (V)')
ax1.set_ylabel('Load (kips)')
ax1.legend(['Python', 'Signal Express'], fontsize=8, loc='upper left')
ax1.grid(True)

# ── Panel 2: Load vs Strain (all 8 channels overlaid) ─────────────────────────
for i in range(8):
    ax2.plot(py_strain[:, i],   py_load, color='blue', linewidth=0.8,
             label='Python' if i == 0 else '_')
    ax2.plot(se_strain_m[:, i], py_load, color='red',  linewidth=0.8, alpha=0.7,
             label='Signal Express' if i == 0 else '_')
ax2.set_title('Load vs Strain')
ax2.set_xlabel('Strain')
ax2.set_ylabel('Load (kips)')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True)

# ── Panel 3: Load vs Pressure (all 4 channels overlaid) ──────────────────────
for i in range(4):
    ax3.plot(py_pressure[:, i], py_load, color='blue', linewidth=0.8,
             label=f'PY {press_short[i]}')
    ax3.plot(se_press_m[:, i],  py_load, color='red',  linewidth=0.8, alpha=0.7,
             label=f'SE {press_short[i]}')
ax3.set_title('Load vs Pressure')
ax3.set_xlabel('Pressure')
ax3.set_ylabel('Load (kips)')
ax3.legend(fontsize=7, loc='upper left')
ax3.grid(True)

# ── Panel 4: Time vs Load (raw recording from each system) ────────────────────
ax4.plot(py_time, py_load, color='blue', linewidth=1,   label='Python')
ax4.plot(se_time, se_load, color='red',  linewidth=1, alpha=0.7, label='Signal Express')
ax4.set_title('Time vs Load')
ax4.set_xlabel('Time')
ax4.set_ylabel('Load (kips)')
ax4.legend(fontsize=8, loc='upper left')
ax4.grid(True)

plt.tight_layout()
plt.savefig('comparison_overview.png', dpi=150)
print("\nSaved: comparison_overview.png")

# ─── DETAILED PER-CHANNEL FIGURES ─────────────────────────────────────────────
# ── Detailed: Load vs Displacement ────────────────────────────────────────────
fig1, axes1 = plt.subplots(4, 3, figsize=(18, 16))
axes1 = axes1.flatten()
for i in range(12):
    ax = axes1[i]
    ax.plot(py_disp[:, i],   py_load, label='Python',         color='blue', linewidth=1)
    ax.plot(se_disp_m[:, i], py_load, label='Signal Express', color='red',  linewidth=1, alpha=0.7)
    ax.set_title(disp_short[i], fontsize=10)
    ax.set_xlabel('Displacement')
    ax.set_ylabel('Load (kips)')
    ax.legend(fontsize=7)
    ax.grid(True)
fig1.suptitle('Load vs Displacement — Python vs Signal Express', fontsize=14)
plt.tight_layout()
plt.savefig('load_vs_displacement.png', dpi=150)
print("Saved: load_vs_displacement.png")

# ── Detailed: Load vs Strain ───────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 10))
axes2 = axes2.flatten()
for i in range(8):
    ax = axes2[i]
    ax.plot(py_strain[:, i],   py_load, label='Python',         color='blue', linewidth=1)
    ax.plot(se_strain_m[:, i], py_load, label='Signal Express', color='red',  linewidth=1, alpha=0.7)
    ax.set_title(strain_short[i], fontsize=10)
    ax.set_xlabel('Strain')
    ax.set_ylabel('Load (kips)')
    ax.legend(fontsize=7)
    ax.grid(True)
fig2.suptitle('Load vs Strain — Python vs Signal Express', fontsize=14)
plt.tight_layout()
plt.savefig('load_vs_strain.png', dpi=150)
print("Saved: load_vs_strain.png")

# ── Detailed: Load vs Pressure ─────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
axes3 = axes3.flatten()
for i in range(4):
    ax = axes3[i]
    ax.plot(py_pressure[:, i], py_load, label='Python',         color='blue', linewidth=1)
    ax.plot(se_press_m[:, i],  py_load, label='Signal Express', color='red',  linewidth=1, alpha=0.7)
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
print("\nDone! All plots saved to your project folder.")