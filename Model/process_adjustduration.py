import os
from moviepy.editor import VideoFileClip
from moviepy.video.fx import speedx

input_folder = "./output/videos_keyframedetection/raw_data"
output_folder = "./output/videos_keyframedetection/adjusted_data"
median_duration = 2.335669002335669  # ความยาวเป็นวินาที

os.makedirs(output_folder, exist_ok=True)

def adjust_video_duration(input_path, output_path, target_duration):
    with VideoFileClip(input_path) as video:
        original_duration = video.duration
        speed_factor = original_duration / target_duration
        adjusted_video = video.fx(speedx.speedx, speed_factor)
        adjusted_video.write_videofile(output_path, codec="libx264")

file_list = [filename for filename in os.listdir(input_folder) if filename.endswith(".mp4")]
total_files = len(file_list)

for i, filename in enumerate(file_list):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    adjust_video_duration(input_path, output_path, median_duration)
    print(f"Processing video {i+1}/{total_files} : {filename}")

print("Video length adjustment is complete.")