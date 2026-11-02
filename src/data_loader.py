"""
Data loader for the UAV GPS Spoofing Detection Dataset
(University of North Dakota, Mendeley z7dj3yyzt8, CC BY 4.0).

Raw file is WIDE: 14 measurement groups x 8 GPS receiver channels = 112 columns,
with a 2-row header. 13 groups are features, the 14th ('Output') is the label
(0=authentic, 1/2/3 = spoofing sophistication levels).

We reshape WIDE -> LONG: each (epoch, active-channel) pair becomes one sample
with 13 features + 1 label. Inactive channels (all-zero rows) are dropped.
This matches how the reference papers reach ~99.5% accuracy.
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
XLSX_PATH = os.path.join(DATASET_DIR, "GPS_Spoofing_UAV.xlsx")
PARQUET_PATH = os.path.join(DATASET_DIR, "gps_spoofing_long.parquet")

# 14 measurement groups in column order; each spans 8 channels (ch0..ch7).
GROUPS = [
    "PRN", "Carrier_Doppler_hz", "Pseudorange_m", "RX_time",
    "TOW_at_current_symbol_s", "Carrier_phase_cycles", "EC", "LC",
    "PC", "PIP", "PQP", "TCD", "CN0", "Output",
]
FEATURE_GROUPS = GROUPS[:13]          # 13 features
N_CHANNELS = 8
CLASS_MAP = {0: "authentic", 1: "simplistic", 2: "intermediate", 3: "sophisticated"}


def _reshape_wide_to_long(raw: np.ndarray) -> pd.DataFrame:
    """raw: (n_epochs, 112) numeric array -> long df (n_active_channel_samples, 13 feats + class + channel)."""
    n_epochs = raw.shape[0]
    blocks = []
    for c in range(N_CHANNELS):
        feat_idx = [g * N_CHANNELS + c for g in range(13)]   # feature col for each group at channel c
        label_idx = 13 * N_CHANNELS + c                       # Output col at channel c
        feats = raw[:, feat_idx]                              # (n_epochs, 13)
        labels = raw[:, label_idx]                            # (n_epochs,)
        block = pd.DataFrame(feats, columns=FEATURE_GROUPS)
        block["Output"] = labels
        block["channel"] = c
        block["epoch_id"] = np.arange(n_epochs)              # original epoch (time) index
        blocks.append(block)
    long_df = pd.concat(blocks, ignore_index=True)
    return long_df


def build_dataset(force_rebuild=False) -> pd.DataFrame:
    if os.path.exists(PARQUET_PATH) and not force_rebuild:
        print(f"Loading cached parquet: {PARQUET_PATH}")
        return pd.read_parquet(PARQUET_PATH)

    print(f"Reading raw xlsx (this takes a minute): {XLSX_PATH}")
    # Skip the 2 header rows; assign our own flat structure afterward.
    raw_df = pd.read_excel(XLSX_PATH, sheet_name="Sheet1", header=None, skiprows=2, engine="openpyxl")
    print(f"Raw shape (epochs x cols): {raw_df.shape}")

    raw = raw_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    if raw.shape[1] != 112:
        print(f"WARNING: expected 112 columns, got {raw.shape[1]}")

    long_df = _reshape_wide_to_long(raw)
    print(f"Long shape (all channel-samples): {long_df.shape}")

    # Drop inactive channels: rows where every feature is 0 (no satellite tracked).
    feat_nonzero = (long_df[FEATURE_GROUPS] != 0).any(axis=1)
    long_df = long_df[feat_nonzero].reset_index(drop=True)
    print(f"After dropping inactive channels: {long_df.shape}")

    # Map numeric label -> class name
    long_df["Output"] = long_df["Output"].round().astype(int)
    long_df = long_df[long_df["Output"].isin(CLASS_MAP.keys())].reset_index(drop=True)
    long_df["class"] = long_df["Output"].map(CLASS_MAP)

    long_df.to_parquet(PARQUET_PATH, index=False)
    print(f"Cached to {PARQUET_PATH}")
    return long_df


def load_data(force_rebuild=False):
    df = build_dataset(force_rebuild=force_rebuild)
    print(f"\nFinal dataset shape: {df.shape}")
    print("Class distribution:")
    print(df["class"].value_counts())
    print("\nPercentages:")
    print((df["class"].value_counts(normalize=True) * 100).round(2))
    return df


if __name__ == "__main__":
    df = load_data(force_rebuild=True)
    print("\nFeature columns:", FEATURE_GROUPS)
    print("\nSample rows:")
    print(df.head())
    print("\nFeature describe:")
    print(df[FEATURE_GROUPS].describe().T[["mean", "std", "min", "max"]])
