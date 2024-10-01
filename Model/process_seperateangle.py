import os
import shutil
import pandas as pd

# Path to the CSV file and video folder
golfdb_path = './golfdb/data/GolfDB.csv'
video_folder = './output/baseline/combined/adjusted/realtime/hpe/csv_aftermodel/predictions/feature correlation/after5fold_cleaned_dtw/'

# Load the CSV file
golfdb = pd.read_csv(golfdb_path)

# Create a dictionary to map view types to folder names
view_folders = {
    'face-on': 'face_on',
    'down-the-line': 'down_the_line',
    'other': 'other'
}

# Create directories for each view type if they don't exist in the video folder
for folder in view_folders.values():
    folder_path = os.path.join(video_folder, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Iterate through the rows in the DataFrame
for index, row in golfdb.iterrows():
    video_id = row['id']
    view = row['view']

    # Determine the folder name based on the view type
    if view in view_folders:
        folder_name = view_folders[view]
    else:
        folder_name = view_folders['other']

    # Source and destination paths
    src_path = os.path.join(video_folder, f'cleaned_dtw_predicted_{video_id}.csv')
    dst_path = os.path.join(video_folder, folder_name, f'cleaned_dtw_predicted_{video_id}.csv')

    # Move the video file to the corresponding folder
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f'Moved {video_id}.csv to {folder_name}')
    else:
        print(f'Video file {video_id}.csv not found.')
