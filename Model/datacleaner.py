import cv2
import mediapipe as mp
import os
import numpy as np

mp_selfie_segmentation = mp.solutions.selfie_segmentation

input_folder = './input/data/videos_160'
output_folder = './output/new_video_cleaned'
os.makedirs(output_folder, exist_ok=True)

def remove_background(image, threshold=0.5, dilation_iterations=5):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    results = selfie_segmentation.process(image_rgb)
    mask = results.segmentation_mask > threshold
    mask = mask.astype(np.uint8)

    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    
    mask_3d = np.stack((mask,) * 3, axis=-1)

    black_background = np.zeros_like(image)
    blurred_background = cv2.GaussianBlur(image, (21, 21), 0)
    dark_background = cv2.addWeighted(image, 0.4, blurred_background, 0.8, 0)
    bg_blurred = np.where(mask_3d, image, blurred_background)
    return bg_blurred

def crop_center(frame):
    height, width, _ = frame.shape
    new_width = width // 2
    start_x = (width - new_width) // 2
    return frame[:, start_x:start_x + new_width]

video_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.mp4')], key=lambda x: int(x.split('.')[0]))
start_idx = video_files.index('0.mp4')

for idx, video_file in enumerate(video_files):
    print(f"Processing video {idx + 1}/{len(video_files)}: {video_file}")
    
    video_path = os.path.join(input_folder, video_file)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(output_folder, f'cleaned_{video_file}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width // 2, original_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = remove_background(frame)
        frame = crop_center(frame)
        out.write(frame)

    cap.release()
    out.release()

print("Processing completed.")