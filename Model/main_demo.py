import cv2
import mediapipe as mp
import os
import random
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_selfie_segmentation = mp.solutions.selfie_segmentation

selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)


input_folder = './input/data/videos_160'
output_folder = './output/Demo_videos'
output_csv = './output/Demo_videos/Demo_csv'

# Read the CSV file and filter for 'face-on' videos
golfdb_path = './golfdb/data/GolfDB.csv'
golfdb = pd.read_csv(golfdb_path)
face_on_videos = golfdb[golfdb['view'] == 'face-on']['id'].tolist()

# Ensure we only select available videos in the input folder
available_videos = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
face_on_video_files = [f"{vid_id}.mp4" for vid_id in face_on_videos if f"{vid_id}.mp4" in available_videos]

# List the IDs of 'face-on' videos
print("List of face-on video IDs:", face_on_videos)

# Select a random 'face-on' video
if not face_on_video_files:
    raise ValueError("No 'face-on' videos found in the input folder.")
video_random = random.sample(face_on_video_files, 1)

global center_x_prev, center_y_prev, radius_prev
center_x_prev = 0
center_y_prev = 0
radius_prev = 0

def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid (The answer is angle of this joint.)
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # if angle > 180.0:
    #     angle = 360 - angle
        
    return angle

def remove_background(image, threshold=0.5, dilation_iterations=5):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    results = selfie_segmentation.process(image_rgb)
    mask = results.segmentation_mask > threshold
    mask = mask.astype(np.uint8)

    # ขยายมาสก์เพื่อเพิ่มพื้นที่รอบๆ ร่างกาย
    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    
    mask_3d = np.stack((mask,) * 3, axis=-1)  # Convert 1-channel mask to 3 channels

    # เบลอพื้นหลัง
    blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
    black_background = np.zeros_like(image)
    black_background[:] = (0, 0, 0)  # สีดำ

    # ทำให้พื้นหลังมีความเข้มของสีลง 60%
    dark_background = cv2.addWeighted(image, 0.4, black_background, 0.8, 0)
    # ผสานภาพที่มีพื้นหลังเบลอกับวัตถุ
    bg_blurred = np.where(mask_3d, image, black_background)
    return bg_blurred

def crop_center(frame):
    # Function to crop the center part of the frame width while keeping the height unchanged
    height, width, _ = frame.shape
    new_width = width // 2
    start_x = (width - new_width) // 2
    return frame[:, start_x:start_x + new_width]

# video_name = '1238.mp4' #specify ไว้เทียบเพื่อดูผลการจูน
video_name = '59.mp4'
# for video_name in video_random:       #comment อันนี้ถ้าจะ ระบุคลิป
#     if video_name.endswith('.mp4'):   #comment อันนี้ถ้าจะ ระบุคลิป
video_path = os.path.join(input_folder, video_name)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
time = np.arange(num_frames) / fps

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = os.path.join(output_folder, f'output_main_{video_name}')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (original_width, original_height))

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    smooth_landmarks=True,
    enable_segmentation=True,
    smooth_segmentation=True,
    refine_face_landmarks=True,
    model_complexity=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6)

# Lists to store angles
arr_left_shoulder = []
arr_right_shoulder = []
arr_left_elbow = []
arr_right_elbow = []
arr_left_hip = []
arr_right_hip = []
arr_left_knee = []
arr_right_knee = []

arr_left_wrist_x = []
arr_left_wrist_y = []
arr_right_wrist_x = []
arr_right_wrist_y = []

csv_filename = f'{video_name}_angles.csv'
csv_path = os.path.join(output_csv, csv_filename)

with open(csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Time', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 
                            'Right Elbow', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee'])

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img_h, img_w, _ = img.shape
        img = remove_background(img)
        img = crop_center(img)
        img_result = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = holistic.process(img)
        img.flags.writeable = True

        mp_drawing.draw_landmarks(
            img_result,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark

                shoulder_l = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_l = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_l = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]

                shoulder_r = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_r = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_r = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
                
                hip_l = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
                knee_l = [landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].y]
                ankle_l = [landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].y]
                
                hip_r = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
                knee_r = [landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_r = [landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].y]
                
                angle_left_shoulder = calculate_angle(elbow_l, shoulder_l, hip_l)
                angle_right_shoulder = calculate_angle(elbow_r, shoulder_r, hip_r)

                angle_left_elbow = calculate_angle(shoulder_l, elbow_l, wrist_l)
                angle_right_elbow = calculate_angle(shoulder_r, elbow_r, wrist_r)

                angle_left_hip = calculate_angle(shoulder_l, hip_l, knee_l)
                angle_right_hip = calculate_angle(shoulder_r, hip_r, knee_r)

                angle_left_knee = calculate_angle(hip_l, knee_l, ankle_l)
                angle_right_knee = calculate_angle(hip_r, knee_r, ankle_r)

                csv_writer.writerow([cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, angle_left_shoulder, angle_right_shoulder, angle_left_elbow, angle_right_elbow, angle_left_hip, angle_right_hip, angle_left_knee, angle_right_knee])

                print("Left Shoulder : " + str(int(angle_left_shoulder)) + "   Right Shoulder : " + str(int(angle_right_shoulder)))
                print("Left Elbow    : " + str(int(angle_left_elbow)) + "   Right Elbow    : " + str(int(angle_right_elbow)))
                print("Left Hip      : " + str(int(angle_left_hip)) + "   Right Hip      : " + str(int(angle_right_hip)))
                print("Left Knee     : " + str(int(angle_left_knee)) + "   Right Knee     : " + str(int(angle_right_knee)) + "\n")
                
                # cv2.putText(
                #     img_result,
                #     "Shoulder L : " + str(int(angle_left_shoulder)) + " R : " + str(int(angle_right_shoulder)),
                #     (img_w - 150, 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.35,
                #     (0, 255, 0),
                #     1,
                #     cv2.LINE_AA
                # )
                # cv2.putText(
                #     img_result,
                #     "Elbow L : " + str(int(angle_left_elbow)) + " R : " + str(int(angle_right_elbow)),
                #     (img_w - 150, 25),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.35,
                #     (0, 0, 255),
                #     1,
                #     cv2.LINE_AA
                # )
                # cv2.putText(
                #     img_result,
                #     "Hip L : " + str(int(angle_left_hip)) + " R : " + str(int(angle_right_hip)),
                #     (img_w - 150, 40),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.35,
                #     (0, 255, 0),
                #     1,
                #     cv2.LINE_AA
                # )
                # cv2.putText(
                #     img_result,
                #     "Knee L : " + str(int(angle_left_knee)) + " R : " + str(int(angle_right_knee)),
                #     (img_w - 150, 55),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.35,
                #     (0, 0, 255),
                #     1,
                #     cv2.LINE_AA
                # )

                arr_left_shoulder.append(angle_left_shoulder)
                arr_right_shoulder.append(angle_right_shoulder)
                arr_left_elbow.append(angle_left_elbow)
                arr_right_elbow.append(angle_right_elbow)
                arr_left_hip.append(angle_left_hip)
                arr_right_hip.append(angle_right_hip)
                arr_left_knee.append(angle_left_knee)
                arr_right_knee.append(angle_right_knee)

                left_wrist_x = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x]
                left_wrist_y = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]

                right_wrist_x = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x]
                right_wrist_y = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]

                arr_left_wrist_x.append(left_wrist_x)
                arr_left_wrist_y.append(left_wrist_y)
                arr_right_wrist_x.append(right_wrist_x)
                arr_right_wrist_y.append(right_wrist_y)


            except Exception as e:
                print(e)
                pass

            left_ear_x = landmarks[mp_holistic.PoseLandmark.LEFT_EAR].x * img_w
            left_ear_y = landmarks[mp_holistic.PoseLandmark.LEFT_EAR].y * img_h
            right_ear_x = landmarks[mp_holistic.PoseLandmark.RIGHT_EAR].x * img_w
            right_ear_y = landmarks[mp_holistic.PoseLandmark.RIGHT_EAR].y * img_h
            center_x = int((left_ear_x + right_ear_x) / 2)
            center_y = int((left_ear_y + right_ear_y) / 2)
            radius = int(abs(left_ear_x - right_ear_x) / 2)
            radius = max(radius, 15)
            # cv2.circle(img_result, center=(center_x, center_y),radius=radius, color=(0, 255, 0), thickness=2)

            if center_x < img_w * 0.45 or center_x > img_w * 0.55 or center_y < img_h * 0.45 or center_y > img_h * 0.55:
                center_x = center_x_prev
                center_y = center_y_prev
                radius = radius_prev
            else:
                center_x_prev = center_x
                center_y_prev = center_y
                radius_prev = radius

        out.write(img_result)
        cv2.imshow('Preview Video', maintain_aspect_ratio_resize(img_result, width=400))

        if cv2.waitKey(1) == ord('q'):
            break


window_length = 41  # Choose an odd number for the window length
polyorder = 3       # Polynomial order

arr_left_shoulder_smooth = savgol_filter(arr_left_shoulder, window_length, polyorder)
arr_right_shoulder_smooth = savgol_filter(arr_right_shoulder, window_length, polyorder)
arr_left_elbow_smooth = savgol_filter(arr_left_elbow, window_length, polyorder)
arr_right_elbow_smooth = savgol_filter(arr_right_elbow, window_length, polyorder)
arr_left_hip_smooth = savgol_filter(arr_left_hip, window_length, polyorder)
arr_right_hip_smooth = savgol_filter(arr_right_hip, window_length, polyorder)
arr_left_knee_smooth = savgol_filter(arr_left_knee, window_length, polyorder)
arr_right_knee_smooth = savgol_filter(arr_right_knee, window_length, polyorder)

time = np.linspace(0, num_frames / fps, len(arr_left_shoulder))

plt.plot(time, arr_left_shoulder_smooth, label='Left shoulder')
plt.plot(time, arr_right_shoulder_smooth, label='Right shoulder')
plt.plot(time, arr_left_elbow_smooth, label='Left Elbow')
plt.plot(time, arr_right_elbow_smooth, label='Right Elbow')
plt.plot(time, arr_left_hip_smooth, label='Left Hip')
plt.plot(time, arr_right_hip_smooth, label='Right Hip')
plt.plot(time, arr_left_knee_smooth, label='Left Knee')
plt.plot(time, arr_right_knee_smooth, label='Right Knee')

plt.xlabel('Time (s)')
plt.ylabel('Angle (Degree)')
plt.ylim(0, 360)
plt.title('All Angles (Smoothed)')
plt.legend()
plt.show()

sigma = 2 # Adjust this value to change the smoothing level
arr_left_wrist_x_smooth = gaussian_filter(arr_left_wrist_x, sigma=sigma)
arr_left_wrist_y_smooth = gaussian_filter(arr_left_wrist_y, sigma=sigma)
arr_right_wrist_x_smooth = gaussian_filter(arr_right_wrist_x, sigma=sigma)
arr_right_wrist_y_smooth = gaussian_filter(arr_right_wrist_y, sigma=sigma)

plt.plot(arr_left_wrist_x_smooth, arr_left_wrist_y_smooth, label='Left Wrist (Smoothed)', color='blue', marker='o', linestyle='-', markersize=5)
plt.plot(arr_right_wrist_x_smooth, arr_right_wrist_y_smooth, label='Right Wrist (Smoothed)', color='red', marker='o', linestyle='-', markersize=5)

plt.scatter(arr_left_wrist_x_smooth[0], arr_left_wrist_y_smooth[0], color='green', label='Start', zorder=5)
# plt.scatter(arr_left_wrist_x_smooth[-1], arr_left_wrist_y_smooth[-1], color='purple', label='End', zorder=5)
plt.scatter(arr_right_wrist_x_smooth[0], arr_right_wrist_y_smooth[0], color='green', zorder=5)
# plt.scatter(arr_right_wrist_x_smooth[-1], arr_right_wrist_y_smooth[-1], color='purple', zorder=5)

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Wrist Positions Over Time (Smoothed)')
plt.legend()
plt.show()

holistic.close()
cap.release()
out.release()
cv2.destroyAllWindows()
