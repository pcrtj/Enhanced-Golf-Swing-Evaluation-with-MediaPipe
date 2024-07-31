import os
import pandas as pd
from scipy.signal import savgol_filter

# Parameters for Savitzky-Golay filter
window_length = 35  # Choose an odd number for the window length
polyorder = 3       # Polynomial order

# Input and output directories
input_dir = './output/videos_keyframedetection/adjusted_data/hpe_adjust_data/csv'
output_dir = './output/videos_keyframedetection/adjusted_data/hpe_adjust_data/smoothed_csv'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all CSV files in the input directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

# Counter for files with insufficient data points
insufficient_data_files = 0

for idx, file in enumerate(csv_files):
    print(f"Processing file {idx + 1}/{len(csv_files)}: {file}")
    file_path = os.path.join(input_dir, file)
    df = pd.read_csv(file_path)
    
    # Flag to check if any column was skipped due to insufficient data points
    insufficient_data = False

    # Apply the Savitzky-Golay filter to each column if the number of data points is sufficient
    columns_to_filter = [
        'Left Shoulder Angle', 'Right Shoulder Angle', 
        'Left Elbow Angle', 'Right Elbow Angle', 
        'Left Hip Angle', 'Right Hip Angle', 
        'Left Knee Angle', 'Right Knee Angle', 
        # 'x, y Left Shoulder', 'x, y Right Shoulder', 
        # 'x, y Left Elbow', 'x, y Right Elbow',
        # 'x, y Left Hip', 'x, y Right Hip', 
        # 'x, y Left Knee', 'x, y Right Knee',
        # 'x, y Left Wrist', 'x, y Right Wrist', 
        # 'x, y Left Ankle', 'x, y Right Ankle', 
        # 'x, y Nose'
    ]
    
    for col in columns_to_filter:
        if len(df[col]) >= window_length:
            df[col] = savgol_filter(df[col], window_length, polyorder)
        else:
            print(f"Skipping {col} for {file} due to insufficient data points")
            insufficient_data = True
    
    if insufficient_data:
        insufficient_data_files += 1

    # Save the smoothed data to a new CSV file in the output directory
    output_file_path = os.path.join(output_dir, file)
    df.to_csv(output_file_path, index=False)

print(f"Smoothing and saving completed. Number of files with insufficient data points: {insufficient_data_files}")

