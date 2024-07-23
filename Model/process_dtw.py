import os
import numpy as np
import pandas as pd
from dtw import dtw

# Define the input and output directories
input_path = './output/videos_keyframedetection/raw_data/hpe_raw_data/spine_raw_csv/'
output_path = './output/videos_keyframedetection/raw_data/hpe_raw_data/dtw_spine_raw_csv/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# List of files
input_files = ['spine_keyframe_0_angles.csv', 'spine_keyframe_1_angles.csv', 'spine_keyframe_2_angles.csv']

# Read CSV files into dataframes
dfs = [pd.read_csv(os.path.join(input_path, file)) for file in input_files]

# Function to split x,y coordinates into separate columns
def split_coordinates(df):
    new_df = pd.DataFrame()
    for column in df.columns:
        new_df[f'{column}_x'] = df[column].apply(lambda x: float(str(x).split(',')[0]))
        new_df[f'{column}_y'] = df[column].apply(lambda x: float(str(x).split(',')[1]))
    return new_df

# Split coordinates in all dataframes
dfs = [split_coordinates(df) for df in dfs]

# Perform DTW and save the aligned data
aligned_dfs = []

# Assuming the first dataframe as the reference for DTW
reference = dfs[0]

for df in dfs:
    aligned_columns = {}
    for column in df.columns:
        ref_series = reference[column].values.flatten()  # Ensure 1-D and float
        target_series = df[column].values.flatten()  # Ensure 1-D and float
        alignment = dtw(ref_series, target_series, keep_internals=True)
        aligned_target_series = np.array([target_series[j] for j in alignment.index2])
        aligned_columns[column] = aligned_target_series
    aligned_df = pd.DataFrame(aligned_columns)
    aligned_dfs.append(aligned_df)

# Save the aligned dataframes to the output directory
for i, aligned_df in enumerate(aligned_dfs):
    aligned_df.to_csv(os.path.join(output_path, f'spine_keyframe_{i}_angles_aligned.csv'), index=False)

print("Alignment and saving completed.")
