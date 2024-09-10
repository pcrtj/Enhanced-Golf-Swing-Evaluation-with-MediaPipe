import os
import shutil
import pandas as pd

# Path to the CSV file and video folder
golfdb_path = './golfdb/data/GolfDB.csv'
video_folder = '../Model/output/baseline/other'

# Load the CSV file
golfdb = pd.read_csv(golfdb_path)

# Define the folder names for slowmotion and realtime
folders = {
    0: 'realtime',
    1: 'slowmotion'
}

# Create directories for slowmotion and realtime if they don't exist
for folder in folders.values():
    folder_path = os.path.join(video_folder, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Iterate through the rows in the DataFrame
for index, row in golfdb.iterrows():
    video_id = row['id']
    slow_type = row['slow']

    # Determine the folder name based on the slow column
    if slow_type in folders:
        folder_name = folders[slow_type]
    else:
        folder_name = 'unknown'  # Just in case there's an unknown type

    # Source and destination paths
    src_path = os.path.join(video_folder, f'{video_id}.mp4')  # Assuming video files are named by id
    dest_path = os.path.join(video_folder, folder_name, f'{video_id}.mp4')

    # Move the video file to the correct folder
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
    else:
        print(f" {video_id} found.")
