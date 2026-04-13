# ─── IMPORT LIBRARIES ─────────────────────────────────────────────────────────
import pandas as pd               # For reading and organizing the text file data
import matplotlib.pyplot as plt   # For plotting graphs
import matplotlib.animation as animation  # For updating plots in real time
import numpy as np                # For math calculations

# ─── FILE PATH ────────────────────────────────────────────────────────────────
# Update this to the exact path where Signal Express saves your .txt file
DATA_FILE = r"D:\Armen_Research\Archana\Soil Pit_Concrete Slab\\trial_12.txt"   # ← Change this to your actual file path

# ─── HOW OFTEN TO UPDATE THE PLOT (in milliseconds) ──────────────────────────
UPDATE_INTERVAL = 1000   # 1000ms = every 1 second
# You can make this smaller (e.g. 500) for faster updates
# or larger (e.g. 2000) if the plot feels too busy

# ─── COLUMN NAMES ─────────────────────────────────────────────────────────────
# These match exactly what Signal Express writes in the header row
TIME_COL = "Time"

DISP_COLS = [
    "Subset Voltage - ai0_DCDT_Right_Slab_A1",
    "Subset Voltage - ai1_DCDT_Right_Slab_A2",
    "Subset Voltage - ai2_DCDT_Right_Slab_A3",
    "Subset Voltage - ai3_DCDT_Right_Slab_B1",
    "Subset Voltage - ai4_DCDT_Right_Slab_B3",
    "Subset Voltage - ai5_DCDT_Left_Slab_B1",
    "Subset Voltage - ai6_DCDT_Left_Slab_B2_Bot",
    "Subset Voltage - ai7_DCDT_Left_Slab_B3",
    "Subset Voltage - ai8_DCDT_Left_Slab_C1",
    "Subset Voltage - ai9_DCDT_Left_Slab_C2",
    "Subset Voltage - ai10_DCDT_Left_Slab_C3",
    "Subset Voltage - ai11_DCDT_Beam_B2_Top"
]

VOLT_COLS = [
    "Subset Voltage - ai17_Soil_Plate_Pressure",
    "Subset Voltage - ai18_Agg_Plate_Pressure",
    "Subset Voltage - ai19_Soil_Pore_Water_Pressure",
    "Subset Voltage - ai20_Agg_Pore_Water_Pressure"
]

STRAIN_COLS = [
    "Subset Strain - ai0_SG_2'E_top",
    "Subset Strain - ai1_SG_3'E_top",
    "Subset Strain - ai2_SG_4'E_top",
    "Subset Strain - ai3_SG_4'E_bot",
    "Subset Strain - ai4_SG_5'E_top",
    "Subset Strain - ai5_SG_5'E_bot",
    "Subset Strain - ai6_SG_6'E_top",
    "Subset Strain - ai7_SG_7'E_top"
]

# Short display names for plot legends (easier to read)
DISP_NAMES  = ['A1R','A2R','A3R','B1R','B3R','B1L','B2L_Bot','B3L','C1L','C2L','C3L','Beam']
VOLT_NAMES  = ['Soil Plate', 'Agg Plate', 'Soil Pore Water', 'Agg Pore Water']
STRAIN_NAMES = ['SG2E_t','SG3E_t','SG4E_t','SG4E_b','SG5E_t','SG5E_b','SG6E_t','SG7E_t']

# ─── PRESSURE CONVERSION FORMULAS ─────────────────────────────────────────────
def process_soil_plate_pressure(raw):
    # Converts raw voltage to Soil Plate Pressure
    return -2.86e-5 * raw**2 + 1.0038 * raw + 0.9331

def process_agg_plate_pressure(raw):
    # Converts raw voltage to Aggregate Plate Pressure
    return -1.34e-4 * raw**2 + 2.5171 * raw - 1.3375

def process_soil_pore_water_pressure(raw):
    # Converts raw voltage to Soil Pore Water Pressure
    return -2.79e-5 * raw**2 + 1.0006 * raw + 0.4014

def process_agg_pore_water_pressure(raw):
    # Converts raw voltage to Aggregate Pore Water Pressure
    return -2.93e-5 * raw**2 + 1.0073 * raw + 0.9260

# ─── TARE VALUES (stored once from first row) ─────────────────────────────────
# These will be set when the first row of data is read
# Subtracting the first reading from all future readings zeroes everything out
tare_values = {}       # Empty dictionary to store first-row values
tare_set    = False    # Flag to track whether tare has been set yet
last_row_count = 0     # Track how many rows have already been plotted

# ─── READ AND PROCESS DATA FROM FILE ──────────────────────────────────────────
def read_data():
    """
    Reads the .txt file, applies tare, and processes voltage to pressure.
    Returns a cleaned dataframe ready for plotting.
    """
    global tare_values, tare_set
    # global means we're using the variables defined outside this function

    try:
        # Read the tab-separated text file
        # sep='\t' tells pandas values are separated by tabs
        # engine='python' handles tricky formatting more reliably
        df = pd.read_csv(DATA_FILE, sep='\t', engine='python')

        # Drop any completely empty rows
        df = df.dropna(how='all')

        # Convert all columns except Time to numbers
        # errors='coerce' turns any bad values into NaN instead of crashing
        for col in df.columns:
            if col != TIME_COL:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows where Time is not a number (sometimes headers repeat)
        df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors='coerce')
        df = df.dropna(subset=[TIME_COL])

        # ── Set tare values from first row ──────────────────────────────────
        if not tare_set and len(df) > 0:
            # Loop through all displacement, voltage and strain columns
            # and store the very first value of each
            for col in DISP_COLS + VOLT_COLS + STRAIN_COLS:
                if col in df.columns:
                    tare_values[col] = df[col].iloc[0]
                    # iloc[0] gets the very first row value
            tare_set = True
            print("Tare values set from first row!")

        # ── Apply tare (subtract first value from all values) ───────────────
        if tare_set:
            for col in DISP_COLS + VOLT_COLS + STRAIN_COLS:
                if col in df.columns:
                    df[col] = df[col] - tare_values[col]

        # ── Convert voltage to pressure using formulas ───────────────────────
        # Apply each formula to its matching voltage column
        df['Soil_Plate_Pressure']      = df[VOLT_COLS[0]].apply(process_soil_plate_pressure)
        df['Agg_Plate_Pressure']       = df[VOLT_COLS[1]].apply(process_agg_plate_pressure)
        df['Soil_Pore_Water_Pressure'] = df[VOLT_COLS[2]].apply(process_soil_pore_water_pressure)
        df['Agg_Pore_Water_Pressure']  = df[VOLT_COLS[3]].apply(process_agg_pore_water_pressure)
        # .apply() runs the formula on every single row of that column

        return df

    except Exception as e:
        # If anything goes wrong reading the file, print the error
        # and return an empty dataframe so the plot doesn't crash
        print(f"Error reading file: {e}")
        return pd.DataFrame()

# ─── PRESSURE COLUMN NAMES (after conversion) ─────────────────────────────────
PRESSURE_COLS = [
    'Soil_Plate_Pressure',
    'Agg_Plate_Pressure',
    'Soil_Pore_Water_Pressure',
    'Agg_Pore_Water_Pressure'
]

# ─── SET UP THE FIGURE AND SUBPLOTS ───────────────────────────────────────────
# Create one large figure with 4 subplots arranged in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
# fig is the whole window
# axes is a 2x2 grid of individual plot areas

ax_strain   = axes[0, 0]   # Top left     → Time vs Strain
ax_disp     = axes[0, 1]   # Top right    → Time vs Displacement
ax_pressure = axes[1, 0]   # Bottom left  → Time vs Pressure
ax_pres_disp= axes[1, 1]   # Bottom right → Pressure vs Displacement (B2_Bot)

fig.suptitle('Real Time Data Visualization', fontsize=14)
# suptitle adds one big title across the top of the whole figure

# ─── ANIMATION UPDATE FUNCTION ────────────────────────────────────────────────
def update(frame):
    """
    This function runs automatically every UPDATE_INTERVAL milliseconds.
    It reads only the new rows since the last update and appends them to the plots.
    frame is just a counter that matplotlib passes in — we don't use it directly
    """
    global last_row_count

    df = read_data()    # Read the latest data from file

    # If file is empty or couldn't be read, skip this update
    if df.empty:
        return

    # Only process rows that are new since last update
    new_df = df.iloc[last_row_count:]
    if new_df.empty:
        return

    new_time = new_df[TIME_COL].values

    # ── Plot 1: Time vs Strain (top left) ───────────────────────────────────
    for col, name in zip(STRAIN_COLS, STRAIN_NAMES):
        if col in new_df.columns:
            ax_strain.plot(new_time, new_df[col].values, label=name if last_row_count == 0 else '_nolegend_', linewidth=1)

    # ── Plot 2: Time vs Displacement (top right) ─────────────────────────────
    for col, name in zip(DISP_COLS, DISP_NAMES):
        if col in new_df.columns:
            ax_disp.plot(new_time, new_df[col].values, label=name if last_row_count == 0 else '_nolegend_', linewidth=1)

    # ── Plot 3: Time vs Pressure (bottom left) ───────────────────────────────
    for col, name in zip(PRESSURE_COLS, VOLT_NAMES):
        if col in new_df.columns:
            ax_pressure.plot(new_time, new_df[col].values, label=name if last_row_count == 0 else '_nolegend_', linewidth=1)

    # ── Plot 4: Pressure vs Displacement B2_Bot (bottom right) ───────────────
    b2_bot_col = "Subset Voltage - ai6_DCDT_Left_Slab_B2_Bot"
    for col, name in zip(PRESSURE_COLS, VOLT_NAMES):
        if col in new_df.columns and b2_bot_col in new_df.columns:
            ax_pres_disp.plot(
                new_df[b2_bot_col].values,
                new_df[col].values,
                label=name if last_row_count == 0 else '_nolegend_',
                linewidth=1
            )

    # Add legends and labels only on first draw
    if last_row_count == 0:
        ax_strain.set_title('Time vs Strain')
        ax_strain.set_xlabel('Time (s)')
        ax_strain.set_ylabel('Strain')
        ax_strain.legend(fontsize=7, loc='upper left')
        ax_strain.grid(True)

        ax_disp.set_title('Time vs Displacement')
        ax_disp.set_xlabel('Time (s)')
        ax_disp.set_ylabel('Displacement')
        ax_disp.legend(fontsize=7, loc='upper left')
        ax_disp.grid(True)

        ax_pressure.set_title('Time vs Pressure')
        ax_pressure.set_xlabel('Time (s)')
        ax_pressure.set_ylabel('Pressure')
        ax_pressure.legend(fontsize=7, loc='upper left')
        ax_pressure.grid(True)

        ax_pres_disp.set_title('Pressure vs Displacement (DCDT_Left_Slab_B2_Bot)')
        ax_pres_disp.set_xlabel('Displacement (B2_Bot)')
        ax_pres_disp.set_ylabel('Pressure')
        ax_pres_disp.legend(fontsize=7, loc='upper left')
        ax_pres_disp.grid(True)

        plt.tight_layout()

    # Rescale all axes to fit the full data range
    for ax in [ax_strain, ax_disp, ax_pressure, ax_pres_disp]:
        ax.relim()
        ax.autoscale_view()

    last_row_count = len(df)   # Update count to include rows just plotted

# ─── START ANIMATION ──────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig,              # The figure to animate
    update,           # The function to call every interval
    interval=UPDATE_INTERVAL,  # How often to update in milliseconds
    cache_frame_data=False     # Don't store old frames to save memory
)
# FuncAnimation automatically calls update() every 1000ms (1 second)
# This is what makes the plots refresh in real time

print("Starting real time visualization...")
print("Start recording in Signal Express now!")
print("Close the plot window to stop.")

plt.show()   # Opens the plot window — stays open until you close it