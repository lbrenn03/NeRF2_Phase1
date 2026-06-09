"""
- Take in raw channel mul dataset and the synthetic channel mul dataset to create 
  a larger mixed dataset.
- Does the same thing with labels to create mixed labels.
"""

import pandas as pd
import os

# features
almost_real = pd.read_csv('group_synt_data/absolute_channel/output/channel_mul_raw.csv')
with open('pilot1_train_index.txt') as f:
    idx_list = [int(line.strip()) for line in f if line.strip()]

real = almost_real.iloc[idx_list]

synth = pd.read_csv('group_synt_data/absolute_channel/output/channel_mul_syn_500.csv')
mixed_X = pd.concat([real, synth], ignore_index=True)

# labels
almost_real_labels = pd.read_csv('group_synt_data/absolute_channel/output/channel_mul_raw_labels.csv')
with open('pilot1_train_index.txt') as f:
    idx_list = [int(line.strip()) for line in f if line.strip()]

real_labels = almost_real_labels.iloc[idx_list]


synth_labels = pd.read_csv('group_synt_data/absolute_channel/output/channel_mul_syn_500_labels.csv')
mixed_y = pd.concat([real_labels, synth_labels], ignore_index=True)

mixed_X.to_csv('channel_mul_mixed_big.csv', index=False, float_format='%.17g')
mixed_y.to_csv('channel_mul_mixed_big_labels.csv', index=False)