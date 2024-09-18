import os
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.video.fx import speedx

def calculate_median_duration(folder_path):
    duration_values = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            last_duration_value = df['Time'].iloc[-1]
            duration_values.append(last_duration_value)

    median_duration = np.median(duration_values)
    mean_duration = np.mean(duration_values)
    
    print(f"Median duration: {median_duration}")
    print(f"Mean duration  : {mean_duration}")
    
    return median_duration

def adjust_video_duration(input_path, output_path, target_duration):
    with VideoFileClip(input_path) as video:
        original_duration = video.duration
        speed_factor = original_duration / target_duration
        adjusted_video = video.fx(speedx.speedx, speed_factor)
        adjusted_video.write_videofile(output_path, codec="libx264")

def process_videos(input_folder, output_folder, median_duration):
    os.makedirs(output_folder, exist_ok=True)
    
    file_list = [filename for filename in os.listdir(input_folder) if filename.endswith(".mp4")]
    total_files = len(file_list)

    for i, filename in enumerate(file_list):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        adjust_video_duration(input_path, output_path, median_duration)
        print(f"Processing video {i+1}/{total_files} : {filename}")
        
    print("Video duration adjustment is complete.")

if __name__ == "__main__":
    csv_folder_path = "./output/videos_raw/csv/combined/realtime"
    input_folder = "./output/baseline/combined"
    output_folder = "./output/baseline/combined/adjusted"

    median_duration = calculate_median_duration(csv_folder_path)

    process_videos(input_folder, output_folder, median_duration)