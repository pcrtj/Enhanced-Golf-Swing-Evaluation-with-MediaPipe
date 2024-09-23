from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import mediapipe as mp
import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


# Constants
UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './uploads'
CSV_FOLDER = './uploads'
MODEL_SAVE_PATH = './model/epoch 30'
BASELINE_FOLDER = './baseline/epoch 50/'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

median_duration = 8.208208208208209

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_PATH, 'lstm_golf_swing_model.h5'))
le = joblib.load(os.path.join(MODEL_SAVE_PATH, 'label_encoder.joblib'))
scaler = joblib.load(os.path.join(MODEL_SAVE_PATH, 'scaler.joblib'))

poses = ['Address', 'Toe-Up', 'Mid-Backswing', 'Top', 'Mid-Downswing', 'Impact', 'Mid-Follow-Through', 'Finish']

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
     
    return angle

def adjust_video_duration(input_path, output_path, target_duration):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_duration = total_frames / fps
    
    speed_factor = original_duration / target_duration
    new_fps = fps * speed_factor
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (int(cap.get(3)), int(cap.get(4))))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()

def process_video(video_path, output_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
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

def prepare_data(df):
    for joint in ['Left Wrist', 'Right Wrist', 'Left Ankle', 'Right Ankle', 'Left Shoulder', 'Right Shoulder', 
                  'Left Elbow', 'Right Elbow', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee']:
        df[[f'{joint} x', f'{joint} y']] = df[f'x, y {joint}'].str.split(', ', expand=True).astype(float)
    
    feature_columns = [
        'Time',
        'Left Shoulder Angle', 'Right Shoulder Angle',
        'Left Elbow Angle', 'Right Elbow Angle',
        'Left Hip Angle', 'Right Hip Angle',
        'Left Knee Angle', 'Right Knee Angle',
        'Left Shoulder x', 'Left Shoulder y',
        'Right Shoulder x', 'Right Shoulder y',
        'Left Elbow x', 'Left Elbow y',
        'Right Elbow x', 'Right Elbow y',
        'Left Hip x', 'Left Hip y',
        'Right Hip x', 'Right Hip y',
        'Left Knee x', 'Left Knee y',
        'Right Knee x', 'Right Knee y',
        'Left Wrist x', 'Left Wrist y',
        'Right Wrist x', 'Right Wrist y',
        'Left Ankle x', 'Left Ankle y',
        'Right Ankle x', 'Right Ankle y'
    ]
    
    return df[feature_columns].values

def predict_poses(file_path):
    df = pd.read_csv(file_path)
    X = prepare_data(df)
    X_scaled = scaler.transform(X)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    predictions = model.predict(X_reshaped)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = le.inverse_transform(predicted_classes)
    
    df['Predicted_Pose'] = predicted_labels
    return df

def load_data(file_path):
    df = pd.read_csv(file_path)
    angle_columns = ['Left Shoulder Angle', 'Right Shoulder Angle', 'Left Elbow Angle', 'Right Elbow Angle',
                     'Left Hip Angle', 'Right Hip Angle', 'Left Knee Angle', 'Right Knee Angle']
    return df[angle_columns + ['Predicted_Pose']]

def split_data_by_pose(data):
    return {pose: data[data['Predicted_Pose'] == pose].drop('Predicted_Pose', axis=1) for pose in poses}

def calculate_manhattan_similarity(baseline, user):
    if baseline.empty or user.empty:
        return None
    manhattan_distance = np.sum(np.abs(baseline.mean() - user.mean()))
    max_possible_distance = len(baseline.columns) * 180  # ปรับเป็น 360 องศา
    similarity = 1 - (manhattan_distance / max_possible_distance)
    return similarity

def calculate_dtw_similarity(baseline, user):
    if baseline.empty or user.empty:
        return None
    
    # แปลงข้อมูลเป็น numpy array
    baseline_array = baseline.values
    user_array = user.values
    
    # คำนวณ DTW distance
    distance, _ = fastdtw(baseline_array, user_array, dist=euclidean)
    
    # คำนวณความคล้ายคลึง (คล้ายกับในบทความ แต่เราใช้ 1 / (1 + distance) เพื่อให้ค่าอยู่ระหว่าง 0 และ 1)
    similarity = 1 / (1 + distance)
    
    return similarity

def assess_similarity(user_csv_path):
    baseline_data = []
    for file in os.listdir(BASELINE_FOLDER):
        if file.endswith('.csv'):
            df = load_data(os.path.join(BASELINE_FOLDER, file))
            baseline_data.append(df)

    user_data = load_data(user_csv_path)

    baseline_poses = split_data_by_pose(pd.concat(baseline_data))
    user_poses = split_data_by_pose(user_data)

    similarities = {}
    for pose in poses:
        if pose in baseline_poses and pose in user_poses:
            similarity = calculate_dtw_similarity(baseline_poses[pose], user_poses[pose])
            if similarity is not None:
                similarities[pose] = similarity

    if similarities:
        total_similarity = sum(similarities.values())
        average_similarity = total_similarity / 8  # หารด้วย 8 ตามจำนวนท่าทั้งหมด
    else:
        average_similarity = None

    return similarities, average_similarity

def main(input_video_path):
    # Create necessary folders
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(CSV_FOLDER, exist_ok=True)

    # Generate output file paths
    filename = os.path.basename(input_video_path)
    adjusted_path = os.path.join(UPLOAD_FOLDER, 'adjusted_' + filename)
    output_video_path = os.path.join(PROCESSED_FOLDER, 'processed_' + filename)
    csv_path = os.path.join(CSV_FOLDER, os.path.splitext(filename)[0] + '.csv')
    prediction_csv_path = os.path.join(CSV_FOLDER, 'predicted_' + os.path.basename(csv_path))

    # Process the video
    print("Adjusting video duration...")
    adjust_video_duration(input_video_path, adjusted_path, median_duration)
    
    print("Processing video...")
    process_video(adjusted_path, output_video_path, csv_path)

    print("Predicting golf swing poses...")
    df_with_predictions = predict_poses(csv_path)
    df_with_predictions.to_csv(prediction_csv_path, index=False)

    print("Assessing similarity...")
    similarities, average_similarity = assess_similarity(prediction_csv_path)

    # Print results
    print("\nResults:")
    print(f"Input Video: {input_video_path}")
    print(f"Processed Video: {output_video_path}")
    print(f"CSV Path: {csv_path}")
    print(f"Prediction CSV Path: {prediction_csv_path}")
    print("\nSimilarities:")
    for pose, similarity in similarities.items():
        print(f"{pose}: {similarity * 100:.2f}%")
    print(f"\nAverage Similarity: {average_similarity * 100:.2f}%")

@app.route('/process_video', methods=['POST'])
def process_video_api():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        username = request.form.get('username')
        if not username:
            return jsonify({'error': 'No username provided'}), 400

        # Create a temporary folder for processing
        temp_folder = os.path.join(UPLOAD_FOLDER, 'temp_' + str(int(time.time())))
        os.makedirs(temp_folder, exist_ok=True)

        input_path = os.path.join(temp_folder, 'input_golf.mp4')
        file.save(input_path)

        adjusted_path = os.path.join(temp_folder, 'adjusted_golf.mp4')
        output_path = os.path.join(temp_folder, 'output_golf.mp4')
        csv_path = os.path.join(temp_folder, 'data.csv')
        prediction_csv_path = os.path.join(temp_folder, 'predicted_data.csv')

        # Process the video
        adjust_video_duration(input_path, adjusted_path, median_duration)
        process_video(adjusted_path, output_path, csv_path)

        # Predict poses and assess similarity
        df_with_predictions = predict_poses(csv_path)
        df_with_predictions.to_csv(prediction_csv_path, index=False)
        similarities, average_similarity = assess_similarity(prediction_csv_path)

        # Prepare the results for API response
        results = {
            'username': username,
            'temp_folder': temp_folder,
            'input_video': 'input_golf.mp4',
            'output_video': 'output_golf.mp4',
            'csv_data': 'data.csv',
            'predicted_data': 'predicted_data.csv',
            'similarities': [round(float(similarities.get(pose, 0)) * 100, 2) for pose in poses],
            'average_similarity': round(float(average_similarity) * 100, 2)
        }

        return jsonify(results)

    return jsonify({'error': 'Invalid file type'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, port=5000)