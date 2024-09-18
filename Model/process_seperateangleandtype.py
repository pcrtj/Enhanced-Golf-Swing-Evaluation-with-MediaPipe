import os
import shutil
import pandas as pd

golfdb_path = './golfdb/data/GolfDB.csv'
video_folder = '../Model/input/data/videos_160/'

golfdb = pd.read_csv(golfdb_path)

view_folders = {
    'face-on': 'face_on',
    'down-the-line': 'down_the_line',
    'other': 'other'
}

speed_folders = {
    0: 'realtime',
    1: 'slowmotion'
}

for view_folder in view_folders.values():
    for speed_folder in speed_folders.values():
        folder_path = os.path.join(video_folder, view_folder, speed_folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

for index, row in golfdb.iterrows():
    video_id = row['id']
    view = row['view']
    slow_type = row['slow']

    if view in view_folders:
        view_folder_name = view_folders[view]
    else:
        view_folder_name = view_folders['other']

    if slow_type in speed_folders:
        speed_folder_name = speed_folders[slow_type]
    else:
        speed_folder_name = 'unknown'

    src_path = os.path.join(video_folder, f'{video_id}.mp4')
    dst_path = os.path.join(video_folder, view_folder_name, speed_folder_name, f'{video_id}.mp4')

    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f'Moved {video_id}.mp4 to {view_folder_name}/{speed_folder_name}')
    else:
        print(f'Video file {video_id}.mp4 not found.')

print("Video processing completed.")