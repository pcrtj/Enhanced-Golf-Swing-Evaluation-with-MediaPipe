import pandas as pd

# Load the CSV file
input_path = './output/videos_keyframedetection/raw_data/hpe_raw_data/spine_cal_csv/spine_keyframe_0_angles.csv'
output_path = './output/videos_keyframedetection/raw_data/hpe_raw_data/spine_cal_csv/spine_keyframe_0_angles_rounded.csv'

# Read the CSV file
df = pd.read_csv(input_path)

# List of columns to round
columns_to_round = [
    'x, y Left Shoulder', 'x, y Right Shoulder', 'x, y Left Elbow', 'x, y Right Elbow', 
    'x, y Left Hip', 'x, y Right Hip', 'x, y Left Knee', 'x, y Right Knee', 
    'x, y Left Wrist', 'x, y Right Wrist', 'x, y Left Ankle', 'x, y Right Ankle', 'x, y Nose'
]

# Round the specified columns to three decimal places
df[columns_to_round] = df[columns_to_round].round(3)

# Save the modified DataFrame to a new CSV file
df.to_csv(output_path, index=False)

print(f"Rounded CSV saved to {output_path}")
