from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import csv
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import time
from flask_cors import CORS
from scipy.spatial.distance import euclidean
from process_cleandata import clean_golf_swing_data_dtw, correct_sequence

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './uploads'
CSV_FOLDER = './uploads'
MODEL_SAVE_PATH = './model/after5fold'
FACE_ON_BASELINE_FOLDER = './baseline/after5fold_cleaned_dtw/face_on'
DOWN_THE_LINE_BASELINE_FOLDER = './baseline/after5fold_cleaned_dtw/down_the_line'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

mean_duration = 8.761422104064687

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_PATH, 'lstm_golf_swing_model.h5'))
le = joblib.load(os.path.join(MODEL_SAVE_PATH, 'label_encoder.joblib'))
scaler = joblib.load(os.path.join(MODEL_SAVE_PATH, 'scaler.joblib'))

correct_sequence = ['Preparation', 'Address', 'Toe-Up', 'Mid-Backswing', 'Top', 'Mid-Downswing', 'Impact', 'Mid-Follow-Through', 'Finish']
analyzed_poses = ['Address', 'Toe-Up', 'Mid-Backswing', 'Top', 'Mid-Downswing', 'Impact', 'Mid-Follow-Through', 'Finish']

angle_columns = ['Left Shoulder Angle', 'Right Shoulder Angle', 'Left Elbow Angle', 'Right Elbow Angle',
                 'Left Hip Angle', 'Right Hip Angle', 'Left Knee Angle', 'Right Knee Angle']

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
    
    # Clean the predicted labels
    cleaned_predictions = clean_golf_swing_data_dtw(predicted_labels)
    
    df['Predicted_Pose'] = cleaned_predictions
    return df

def load_data(file_path):
    df = pd.read_csv(file_path)
    angle_columns = ['Left Shoulder Angle', 'Right Shoulder Angle', 'Left Elbow Angle', 'Right Elbow Angle',
                     'Left Hip Angle', 'Right Hip Angle', 'Left Knee Angle', 'Right Knee Angle']
    return df[angle_columns + ['Predicted_Pose']]

def split_data_by_pose(data):
    return {pose: data[data['Predicted_Pose'] == pose].drop('Predicted_Pose', axis=1) for pose in analyzed_poses}

def calculate_similarity(baseline, user):
    if baseline.empty or user.empty:
        return None, 0
    
    baseline_data = baseline[angle_columns].values
    user_data = user[angle_columns].values

    def safe_normalize(data):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        return (data - min_vals) / range_vals

    baseline_norm = safe_normalize(baseline_data)
    user_norm = safe_normalize(user_data)

    distances = []
    for b, u in zip(baseline_norm, user_norm):
        valid_indices = np.isfinite(b) & np.isfinite(u)
        if np.any(valid_indices):
            distances.append(euclidean(b[valid_indices], u[valid_indices]))
        else:
            distances.append(0)

    max_possible_distance = np.sqrt(len(angle_columns))
    similarities = [1 - (d / max_possible_distance) for d in distances]

    angle_similarities = {}
    for i, column in enumerate(angle_columns):
        angle_similarities[column] = np.mean([1 - abs(b[i] - u[i]) for b, u in zip(baseline_norm, user_norm) if np.isfinite(b[i]) and np.isfinite(u[i])])

    overall_similarity = np.mean(similarities)

    return angle_similarities, overall_similarity

def assess_similarity(user_csv_path, face_on_folder=FACE_ON_BASELINE_FOLDER, down_the_line_folder=DOWN_THE_LINE_BASELINE_FOLDER):
    def load_baseline_data(folder):
        baseline_data = []
        for file in os.listdir(folder):
            if file.endswith('.csv'):
                try:
                    df = load_data(os.path.join(folder, file))
                    baseline_data.append(df)
                except Exception as e:
                    print(f"Error loading baseline file {file} from {folder}: {e}")
        return baseline_data

    face_on_data = load_baseline_data(face_on_folder)
    down_the_line_data = load_baseline_data(down_the_line_folder)

    if not face_on_data and not down_the_line_data:
        print(f"No baseline data found in {face_on_folder} or {down_the_line_folder}")
        return {}

    try:
        user_data = load_data(user_csv_path)
    except Exception as e:
        print(f"Error loading user data from {user_csv_path}: {e}")
        return {}

    def process_baseline(baseline_data):
        if not baseline_data:
            return {}
        baseline_poses = split_data_by_pose(pd.concat(baseline_data))
        user_poses = split_data_by_pose(user_data)
        similarities = {}
        for pose in analyzed_poses:
            if pose in baseline_poses and pose in user_poses:
                angle_similarities, overall_similarity = calculate_similarity(baseline_poses[pose], user_poses[pose])
                similarities[pose] = (angle_similarities, overall_similarity)
        return similarities

    face_on_similarities = process_baseline(face_on_data)
    down_the_line_similarities = process_baseline(down_the_line_data)

    best_similarities = {}
    for pose in analyzed_poses:
        face_on = face_on_similarities.get(pose, (None, 0))
        down_the_line = down_the_line_similarities.get(pose, (None, 0))
        best_similarities[pose] = max([face_on, down_the_line], key=lambda x: x[1] if x[1] is not None else 0)

    return best_similarities

def main(input_video_path):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(CSV_FOLDER, exist_ok=True)

    filename = os.path.basename(input_video_path)
    adjusted_path = os.path.join(UPLOAD_FOLDER, 'adjusted_' + filename)
    output_video_path = os.path.join(PROCESSED_FOLDER, 'processed_' + filename)
    csv_path = os.path.join(CSV_FOLDER, os.path.splitext(filename)[0] + '.csv')
    prediction_csv_path = os.path.join(CSV_FOLDER, 'predicted_' + os.path.basename(csv_path))

    print("Adjusting video duration...")
    adjust_video_duration(input_video_path, adjusted_path, mean_duration)
    
    print("Processing video...")
    process_video(adjusted_path, output_video_path, csv_path)

    print("Predicting golf swing poses...")
    df_with_predictions = predict_poses(csv_path)
    df_with_predictions.to_csv(prediction_csv_path, index=False)

    cleaned_csv_path = os.path.join(CSV_FOLDER, 'cleaned_predicted_' + os.path.basename(csv_path))
    df_with_predictions.to_csv(cleaned_csv_path, index=False)

    print("Assessing similarity...")
    similarities = assess_similarity(cleaned_csv_path)

    print("\nResults:")
    print(f"Input Video: {input_video_path}")
    print(f"Processed Video: {output_video_path}")
    print(f"CSV Path: {csv_path}")
    print(f"Prediction CSV Path: {prediction_csv_path}")
    print(f"Cleaned Prediction CSV Path: {cleaned_csv_path}")
    print("\nSimilarities:")
    for pose in analyzed_poses:
        if pose in similarities:
            angle_similarities, overall_similarity = similarities[pose]
            print(f"{pose}: {overall_similarity * 100:.2f}%")
    
    average_similarity = np.mean([similarities[pose][1] for pose in analyzed_poses if pose in similarities and similarities[pose][1] is not None])
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

        temp_folder = os.path.join(UPLOAD_FOLDER, 'temp_' + str(int(time.time())))
        os.makedirs(temp_folder, exist_ok=True)

        input_path = os.path.join(temp_folder, 'input_golf.mp4')
        file.save(input_path)

        adjusted_path = os.path.join(temp_folder, 'adjusted_golf.mp4')
        output_path = os.path.join(temp_folder, 'output_golf.mp4')
        csv_path = os.path.join(temp_folder, 'data.csv')

        adjust_video_duration(input_path, adjusted_path, mean_duration)
        process_video(adjusted_path, output_path, csv_path)

        df_with_predictions = predict_poses(csv_path)
    
        # บันทึก predicted_data
        predicted_csv_path = os.path.join(temp_folder, 'predicted_data.csv')
        df_with_predictions.to_csv(predicted_csv_path, index=False)
    
        # บันทึก cleaned_predicted_data
        cleaned_csv_path = os.path.join(temp_folder, 'cleaned_predicted_data.csv')
        df_with_predictions.to_csv(cleaned_csv_path, index=False)

        # ตรวจสอบว่าไฟล์ถูกสร้างขึ้นจริง
        if os.path.exists(cleaned_csv_path):
            print(f"File {cleaned_csv_path} was created successfully.")
        else:
            print(f"Failed to create file {cleaned_csv_path}.")
        
        similarities = assess_similarity(cleaned_csv_path)
    
        if not similarities:
            return jsonify({'error': 'Failed to calculate similarities'}), 500

        similarity_data = []
        for pose in analyzed_poses:
            if pose in similarities:
                angle_similarities, overall_similarity = similarities[pose]
                if angle_similarities is not None:
                    pose_data = [min(100, round(angle_similarities.get(angle, 0) * 100 * 1.25, 2)) for angle in angle_columns]
                else:
                    pose_data = [0] * len(angle_columns)
                similarity_data.append(pose_data)
            else:
                similarity_data.append([0] * len(angle_columns))

        if not similarity_data:
            return jsonify({'error': 'No similarity data available'}), 500

        average_similarity = sum([sum(pose_data) / len(pose_data) for pose_data in similarity_data]) / len(similarity_data)

        # อ่านเนื้อหาของไฟล์ CSV
        with open(predicted_csv_path, 'r') as f:
            predicted_data = f.read()
    
        with open(cleaned_csv_path, 'r') as f:
            cleaned_predicted_data = f.read()

        results = {
            'username': username,
            'temp_folder': temp_folder,
            'input_video': 'input_golf.mp4',
            'output_video': 'output_golf.mp4',
            'csv_data': 'data.csv',
            'predicted_data': predicted_data,
            'cleaned_predicted_data': cleaned_predicted_data,
            'similarities': similarity_data,
            'average_similarity': min(100, round(average_similarity, 2))
        }

        return jsonify(results)

    return jsonify({'error': 'Invalid file type'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, port=5000)