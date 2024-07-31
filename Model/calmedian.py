import os
import pandas as pd
import numpy as np

folder_path = "./output/videos_keyframedetection/raw_data/hpe_raw_data/keyframe_raw_csv"

time_values = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        last_time_value = df['Time'].iloc[-1]
        time_values.append(last_time_value)

median_time = np.median(time_values)
mean_time = np.mean(time_values)

print(f"Median of Time values: {median_time}")
print(f"Mean of Time values  : {mean_time}")
