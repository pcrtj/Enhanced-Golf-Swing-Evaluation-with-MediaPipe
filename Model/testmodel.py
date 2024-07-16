import cv2
import mediapipe as mp
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_selfie_segmentation = mp.solutions.selfie_segmentation

input_folder = './input/data/videos_160'
output_folder = 'output'
output_csv = 'output/csv'

# Read the CSV file and filter for 'face-on' videos
golfdb_path = './golfdb/data/GolfDB.csv'
golfdb = pd.read_csv(golfdb_path)
face_on_videos = golfdb[golfdb['view'] == 'face-on']['id'].tolist()

# Ensure we only select available videos in the input folder
available_videos = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
face_on_video_files = [f"{vid_id}.mp4" for vid_id in face_on_videos if f"{vid_id}.mp4" in available_videos]

# List the IDs of 'face-on' videos
print("List of face-on video IDs:", face_on_videos)

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid (The angle is at this joint)
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def extract_angles_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    angles_list = []
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Example: Calculate angle for the left elbow
                left_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, 
                                 landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x, 
                              landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x, 
                              landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
                
                angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                angles_list.append(angle)
                
    cap.release()
    return angles_list

def remove_background(image, threshold=0.5, dilation_iterations=5):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    results = selfie_segmentation.process(image_rgb)
    mask = results.segmentation_mask > threshold
    mask = mask.astype(np.uint8)

    # ขยายมาสก์เพื่อเพิ่มพื้นที่รอบๆ ร่างกาย
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    
    mask_3d = np.stack((mask,) * 3, axis=-1)  # Convert 1-channel mask to 3 channels

    # เบลอพื้นหลัง
    blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
    
    # ผสานภาพที่มีพื้นหลังเบลอกับวัตถุ
    bg_blurred = np.where(mask_3d, image, blurred_image)
    return bg_blurred
# Extract angles for each video and save to CSV
all_angles = []

for idx, video_file in enumerate(face_on_video_files):
    print(f'Processing video {idx + 1}/{len(face_on_video_files)}: {video_file}')
    video_path = os.path.join(input_folder, video_file)
    video_path = remove_background(video_path)
    angles = extract_angles_from_video(video_path)
    all_angles.extend(angles)
    
# Split data into training and testing sets
X = np.array(all_angles).reshape(-1, 1)
y = np.array([1 if i % 2 == 0 else 0 for i in range(len(X))])  # Example labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'golf_swing_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f'Model saved to {model_filename}')

# Load the trained model from a file
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Test the loaded model
y_pred_loaded = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)
print(f'Loaded model accuracy: {loaded_accuracy * 100:.2f}%')
