import cv2
import os
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip

def remove_background(image, threshold=0.5, dilation_iterations=5):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    results = selfie_segmentation.process(image_rgb)
    mask = results.segmentation_mask > threshold
    mask = mask.astype(np.uint8)

    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    
    mask_3d = np.stack((mask,) * 3, axis=-1)

    blurred_background = cv2.GaussianBlur(image, (21, 21), 0)
    bg_blurred = np.where(mask_3d, image, blurred_background)
    return bg_blurred

def process_video(input_path, output_path):
    video_clip = VideoFileClip(input_path)
    output_frames = []

    for frame in video_clip.iter_frames():
        processed_frame = remove_background(frame)
        output_frames.append(processed_frame)

    processed_clip = VideoFileClip(np.array(output_frames), fps=video_clip.fps)
    processed_clip.write_videofile(output_path, codec='libx264')

def process_videos(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4") or f.endswith(".avi")]
    # start_idx = video_files.index('keyframe_0.mp4')

    for idx, video_file in enumerate(video_files):
        print(f"Processing video {idx + 1}/{len(video_files)}: {video_file}")
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, f'blurback_{video_file}')
        process_video(input_path, output_path)

if __name__ == "__main__":
    input_dir = "./output/videos_keyframedetection/raw_data"
    output_dir = "./output/videos_keyframedetection/raw_data/blurback_raw_data"
    process_videos(input_dir, output_dir)
