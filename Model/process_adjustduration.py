import os
from moviepy.editor import VideoFileClip
from moviepy.video.fx import speedx

input_folder = "./output/baseline/combined"
output_folder = "./output/baseline/combined/adjusted"
# median_duration = 2.335669002335669  # ความยาวเป็นวินาที
# median_duration = 9.40940940940941  # down-the-line
# median_duration = 9.65006010418058  # face-on
# median_duration = 9.107584346632601  # other
# median_duration = 9.540647925555941  # combined
median_duration = 8.208208208208209  # combined realtime
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