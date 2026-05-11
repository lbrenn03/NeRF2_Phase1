import pandas as pd
import numpy as np

# Load files properly (keep headers)
rssi_df = pd.read_csv("gateway_rssi.csv")   # g0–g5 header
rx_df = pd.read_csv("rx_pos.csv")           # x,y,z header

# Replace -100 with NaN so max works correctly
rssi_df = rssi_df.replace(-100, np.nan)

# Combine rx_pos + rssi
combined = pd.concat([rx_df, rssi_df], axis=1)

# Group by rx position columns (x,y,z)
group_cols = rx_df.columns.tolist()

# Merge duplicates
result = combined.groupby(group_cols, as_index=False).max()

# Split back out
combined_rx_pos = result[group_cols]                 # keeps x,y,z header
combined_rssi = result[rssi_df.columns].fillna(-100) # keeps g0–g5 header

# Save
combined_rx_pos.to_csv("rx_pos_combined.csv", index=False)
combined_rssi.to_csv("gateway_rssi_combined.csv", index=False)