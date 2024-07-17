import cv2
import mediapipe as mp
import numpy as np
import os
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

input_folder = './output/videos_keyframedetection/raw_data'
output_folder = './output/videos_keyframedetection/raw_data/hpe_raw_data'
csv_output_folder = './output/videos_keyframedetection/raw_data/hpe_raw_data/csv'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(csv_output_folder, exist_ok=True)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # if angle > 180.0:
    #     angle = 360 - angle
        
    return angle

def process_video(video_path, output_path, csv_folder):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    csv_filename = os.path.splitext(os.path.basename(video_path))[0] + '_angles.csv'
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




video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
# start_idx = video_files.index('0.mp4')

for idx, video_file in enumerate(video_files):
    # print(f"Processing video {idx + 1}/{len(video_files) - start_idx}: {video_file}")
    print(f"Processing video {idx + 1}/{len(video_files)}: {video_file}")
    video_path = os.path.join(input_folder, video_file)
    output_path = os.path.join(output_folder, f'hpe_{video_file}')
    process_video(video_path, output_path, csv_output_folder)

print('Processing complete.')