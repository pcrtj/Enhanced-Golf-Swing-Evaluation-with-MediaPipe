import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

input_path = './output/videos_keyframedetection/raw_data/hpe_raw_data/spine_smooth_csv'
output_path = './output/videos_keyframedetection/raw_data/hpe_raw_data/dtw_spine_raw_csv/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ฟังก์ชันในการอ่าน CSV และใช้ DTW
def process_dtw(file_name):
    # อ่านข้อมูลจากไฟล์
    file_path = os.path.join(input_path, file_name)
    data = pd.read_csv(file_path)
    
    reference_time = data["Time"].values
    aligned_data = {"Time": reference_time}

    columns_of_interest = [
    # "Time",
    "Left Shoulder Angle",
    "Right Shoulder Angle",
    "Left Elbow Angle",
    "Right Elbow Angle",
    "Left Hip Angle",
    "Right Hip Angle",
    "Left Knee Angle",
    "Right Knee Angle"
]   
    # ตรวจสอบว่ามีคอลัมน์ "Time" หรือไม่
    if "Time" not in data.columns:
        print(f"File {file_name} does not contain 'Time' column.")
        return
    # ตรวจสอบว่ามีคอลัมน์ที่สนใจทั้งหมดหรือไม่
    missing_columns = [col for col in columns_of_interest if col not in data.columns]
    if missing_columns:
        print(f"File {file_name} is missing columns: {', '.join(missing_columns)}")
        return
    
    # การใช้ DTW บนคอลัมน์ "Time"
    

    for column in columns_of_interest:
        column_data = data[column].values
        distance, path = fastdtw(reference_time, column_data, dist=2)
        aligned_column = np.array([column_data[j] for i, j in path])
        
        # Check if aligned_column is longer than reference_time
        if aligned_column.shape[0] > reference_time.shape[0]:
            aligned_column = aligned_column[:reference_time.shape[0]]
        else:
            aligned_column = np.pad(aligned_column, (0, reference_time.shape[0] - aligned_column.shape[0]), mode='constant')
        
        aligned_data[column] = aligned_column
        
    # การสร้าง DataFrame ใหม่ที่ปรับแต่งแล้ว
    dtw_data = pd.DataFrame(aligned_data)
    output_file_name = f"dtw_{file_name}"
    output_file_path = os.path.join(output_path, output_file_name)
    dtw_data.to_csv(output_file_path, index=False)
    print(f"Processed {file_name} and saved to {output_file_path}")

# ประมวลผลไฟล์ CSV ทั้งหมดใน input path
for file_name in os.listdir(input_path):
    if file_name.endswith('.csv'):
        process_dtw(file_name)