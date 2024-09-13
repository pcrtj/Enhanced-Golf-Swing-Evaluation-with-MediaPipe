from flask import Flask, request, jsonify
import os
import csv
import pickle
import pandas as pd
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from moviepy.video.fx import speedx
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def adjust_video_duration(input_path, output_path, target_duration=8.208208208208209):
    with VideoFileClip(input_path) as video:
        original_duration = video.duration
        speed_factor = original_duration / target_duration
        adjusted_video = video.fx(speedx.speedx, speed_factor)
        adjusted_video.write_videofile(output_path, codec="libx264")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle

def process_video(video_path, output_path, csv_folder):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    csv_filename = os.path.splitext(os.path.basename(video_path))[0] + '.csv'
    csv_path = os.path.join(csv_folder, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Time', 'Left Shoulder Angle', 'Right Shoulder Angle', 'Left Elbow Angle', 
                             'Right Elbow Angle', 'Left Hip Angle', 'Right Hip Angle', 'Left Knee Angle', 'Right Knee Angle',
                             'x, y Left Shoulder', 'x, y Right Shoulder', 'x, y Left Elbow', 'x, y Right Elbow',
                             'x, y Left Hip', 'x, y Right Hip', 'x, y Left Knee', 'x, y Right Knee',
                             'x, y Left Wrist', 'x, y Right Wrist', 'x, y Left Ankle', 'x, y Right Ankle', 'x, y Nose'])

        with mp_holistic.Holistic(
                static_image_mode=False,
                smooth_landmarks=True,
                enable_segmentation=True,
                smooth_segmentation=True,
                refine_face_landmarks=True,
                model_complexity=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    left_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, 
                                     landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x, 
                                  landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x, 
                                  landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
                    
                    right_shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, 
                                      landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x, 
                                   landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x, 
                                   landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
                    
                    left_hip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x, 
                                landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].x, 
                                 landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    right_hip = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x, 
                                 landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x, 
                                  landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].y]
                    
                    nose = [landmarks[mp_holistic.PoseLandmark.NOSE.value].x, 
                            landmarks[mp_holistic.PoseLandmark.NOSE.value].y]
                    
                    angle_left_shoulder = calculate_angle(left_elbow, left_shoulder, left_hip)
                    angle_right_shoulder = calculate_angle(right_elbow, right_shoulder, right_hip)

                    angle_left_elbow = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    angle_right_elbow = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
                    angle_right_hip = calculate_angle(right_shoulder, right_hip, right_knee)

                    angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
                    angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)

                    csv_writer.writerow([cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, angle_left_shoulder, angle_right_shoulder, 
                                         angle_left_elbow, angle_right_elbow, angle_left_hip, angle_right_hip, angle_left_knee, angle_right_knee,
                                         f"{left_shoulder[0]:.3f}, {left_shoulder[1]:.3f}",
                                         f"{right_shoulder[0]:.3f}, {right_shoulder[1]:.3f}",
                                         f"{left_elbow[0]:.3f}, {left_elbow[1]:.3f}",
                                         f"{right_elbow[0]:.3f}, {right_elbow[1]:.3f}",
                                         f"{left_hip[0]:.3f}, {left_hip[1]:.3f}",
                                         f"{right_hip[0]:.3f}, {right_hip[1]:.3f}",
                                         f"{left_knee[0]:.3f}, {left_knee[1]:.3f}",
                                         f"{right_knee[0]:.3f}, {right_knee[1]:.3f}",
                                         f"{left_wrist[0]:.3f}, {left_wrist[1]:.3f}",
                                         f"{right_wrist[0]:.3f}, {right_wrist[1]:.3f}",
                                         f"{left_ankle[0]:.3f}, {left_ankle[1]:.3f}",
                                         f"{right_ankle[0]:.3f}, {right_ankle[1]:.3f}",
                                         f"{nose[0]:.3f}, {nose[1]:.3f}"])

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                out.write(frame)
    
    cap.release()
    out.release()

def load_model():
    with open('./output/videos_raw/model/combined/random_forest_classifier.pkl', 'rb') as model_file:
        return pickle.load(model_file)

def predict_pose(csv_file, output_csv_path):
    model = load_model()
    df = pd.read_csv(csv_file)

    df[['Left Wrist x', 'Left Wrist y']] = df['x, y Left Wrist'].str.split(', ', expand=True).astype(float)
    df[['Right Wrist x', 'Right Wrist y']] = df['x, y Right Wrist'].str.split(', ', expand=True).astype(float)
    df[['Left Ankle x', 'Left Ankle y']] = df['x, y Left Ankle'].str.split(', ', expand=True).astype(float)
    df[['Right Ankle x', 'Right Ankle y']] = df['x, y Right Ankle'].str.split(', ', expand=True).astype(float)
    
    feature_columns = [
        'Left Shoulder Angle', 'Right Shoulder Angle', 'Left Elbow Angle', 'Right Elbow Angle',
        'Left Hip Angle', 'Right Hip Angle', 'Left Knee Angle', 'Right Knee Angle',
        'Left Wrist x', 'Left Wrist y', 'Right Wrist x', 'Right Wrist y', 
        'Left Ankle x', 'Left Ankle y', 'Right Ankle x', 'Right Ankle y'
    ]
    X = df[feature_columns]

    predictions = model.predict(X)
    df['Pose'] = predictions

    pose_labels = {0: 'Preparation', 1: 'Address', 2: 'Toe-Up', 3: 'Mid-Backswing', 4: 'Top',
                   5: 'Mid-Downswing', 6: 'Impact', 7: 'Mid-Follow-Through', 8: 'Finish'}
    
    df['Predicted Pose Name'] = df['Pose'].map(pose_labels)
    
    output_file = os.path.join(output_csv_path, os.path.basename(csv_file))
    df.to_csv(output_file, index=False)
    return output_file

def compare_with_dataset(user_csv, dataset_folder='./output/baseline/combined/adjusted/realtime/hpe/csv_aftermodel'):
    # โหลดข้อมูลผู้ใช้
    user_df = pd.read_csv(user_csv)
    
    # เตรียมฟีเจอร์สำหรับข้อมูลผู้ใช้
    user_df[['Left Wrist x', 'Left Wrist y']] = user_df['x, y Left Wrist'].str.split(', ', expand=True).astype(float)
    user_df[['Right Wrist x', 'Right Wrist y']] = user_df['x, y Right Wrist'].str.split(', ', expand=True).astype(float)
    user_df[['Left Ankle x', 'Left Ankle y']] = user_df['x, y Left Ankle'].str.split(', ', expand=True).astype(float)
    user_df[['Right Ankle x', 'Right Ankle y']] = user_df['x, y Right Ankle'].str.split(', ', expand=True).astype(float)
    
    feature_columns = [
        'Left Shoulder Angle', 'Right Shoulder Angle', 'Left Elbow Angle', 'Right Elbow Angle',
        'Left Hip Angle', 'Right Hip Angle', 'Left Knee Angle', 'Right Knee Angle',
        'Left Wrist x', 'Left Wrist y', 'Right Wrist x', 'Right Wrist y', 
        'Left Ankle x', 'Left Ankle y', 'Right Ankle x', 'Right Ankle y'
    ]
    user_features = user_df[feature_columns]

    # คำนวณ Cosine Similarity กับทุกไฟล์ในชุดข้อมูล
    similarities = []
    for file in os.listdir(dataset_folder):
        if file.endswith('.csv'):
            dataset_csv = os.path.join(dataset_folder, file)
            dataset_df = pd.read_csv(dataset_csv)
            
            # เตรียมฟีเจอร์สำหรับชุดข้อมูล
            dataset_df[['Left Wrist x', 'Left Wrist y']] = dataset_df['x, y Left Wrist'].str.split(', ', expand=True).astype(float)
            dataset_df[['Right Wrist x', 'Right Wrist y']] = dataset_df['x, y Right Wrist'].str.split(', ', expand=True).astype(float)
            dataset_df[['Left Ankle x', 'Left Ankle y']] = dataset_df['x, y Left Ankle'].str.split(', ', expand=True).astype(float)
            dataset_df[['Right Ankle x', 'Right Ankle y']] = dataset_df['x, y Right Ankle'].str.split(', ', expand=True).astype(float)
            
            dataset_features = dataset_df[feature_columns]
            
            # คำนวณความคล้ายคลึงกัน
            user_features_array = user_features.values
            dataset_features_array = dataset_features.values
            
            # คำนวณความคล้ายคลึงกันสำหรับแต่ละแถว
            similarity_scores = []
            for i in range(dataset_features_array.shape[0]):
                dataset_row = dataset_features_array[i].reshape(1, -1)
                similarities = cosine_similarity(user_features_array, dataset_row)
                similarity_scores.append(similarities.mean())
            
            average_similarity = np.mean(similarity_scores)
            similarities.append((file, average_similarity))
    
    # เรียงลำดับตามความคล้ายคลึงกัน
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Adjust video duration
    output_path = os.path.join('processed_videos', file.filename)
    adjust_video_duration(filepath, output_path)

    # Process the video and save CSV
    csv_output_path = 'output/csv/'
    process_video(output_path, csv_output_path, csv_output_path)

    # Predict pose and save the predicted CSV
    predicted_csv_path = 'output/predicted_csv/'
    predicted_csv = predict_pose(csv_output_path, predicted_csv_path)
    
    # Compare with dataset
    dataset_folder = './output/baseline/combined/adjusted/realtime/hpe/csv_aftermodel'
    similarities = compare_with_dataset(predicted_csv, dataset_folder)

    return jsonify({
        'input_video': filepath,
        'output_video': output_path,
        'csv_output': predicted_csv,
        'similarities': similarities
    })

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('processed_videos'):
        os.makedirs('processed_videos')
    if not os.path.exists('output/csv'):
        os.makedirs('output/csv')
    if not os.path.exists('output/predicted_csv'):
        os.makedirs('output/predicted_csv')
    app.run(debug=True)
