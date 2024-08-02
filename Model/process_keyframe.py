import os
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip

csv_path = './golfdb/data/GolfDB.csv'

input_videos_path = './input/data/videos_160'

output_videos_path = './output/videos_keyframedetection/raw_data'

os.makedirs(output_videos_path, exist_ok=True)

df = pd.read_csv(csv_path)

def cut_video(video_name, start_frame, end_frame, fps, output_path):
    video_path = os.path.join(input_videos_path, video_name)
    with VideoFileClip(video_path) as video:
        start_time = start_frame / fps
        end_time = end_frame / fps
        cut_video = video.subclip(start_time, end_time)
        output_file = os.path.join(output_path, f'keyframe_{video_name}')
        cut_video.write_videofile(output_file, codec='libx264')


for index, row in df.iterrows():
    # video_name = f"{row['id']}.mp4"
    video_name = f"keyframe_{row['id']}.mp4"
    events = list(map(int, row['events'].strip('[]').split(',')))
    # Assuming the frame rate (fps) is 30 for all videos
    fps = 30
    # Print status
    print(f"Processing video {index + 1}/{len(df)}: {video_name}")
    cut_video(video_name, events[1], events[-2], fps, output_videos_path)

print("*** All videos have been processed! ***")
