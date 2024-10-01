import os
import pandas as pd
import numpy as np
from dtaidistance import dtw

# กำหนดลำดับท่าทางที่ถูกต้อง
correct_sequence = ['Preparation', 'Address', 'Toe-Up', 'Mid-Backswing', 'Top', 'Mid-Downswing', 'Impact', 'Mid-Follow-Through', 'Finish']

def clean_golf_swing_data_dtw(predictions, window_size=3):
    # แปลงท่าทางเป็นตัวเลข
    pose_to_num = {pose: i for i, pose in enumerate(correct_sequence)}
    num_to_pose = {i: pose for i, pose in enumerate(correct_sequence)}
    
    # สร้างลำดับตัวเลขจากท่าทางที่ทำนาย
    predicted_sequence = [pose_to_num[pose] for pose in predictions]
    
    # สร้างลำดับตัวเลขจากท่าทางที่ถูกต้อง
    template_sequence = list(range(len(correct_sequence)))
    
    # ใช้ DTW เพื่อหาการจับคู่ที่ดีที่สุด
    alignment = dtw.warping_path(predicted_sequence, template_sequence)
    
    # สร้างลำดับท่าทางที่ clean แล้ว
    cleaned_sequence = [template_sequence[j] for i, j in alignment]
    
    # ใช้ custom median filter เพื่อลด noise
    def custom_median_filter(sequence, window_size):
        result = []
        for i in range(len(sequence)):
            start = max(0, i - window_size // 2)
            end = min(len(sequence), i + window_size // 2 + 1)
            window = sequence[start:end]
            median = int(np.median(window))
            result.append(median)
        return result

    cleaned_sequence = custom_median_filter(cleaned_sequence, window_size)
    
    # แปลงกลับเป็นชื่อท่าทาง
    cleaned_predictions = [num_to_pose[num] for num in cleaned_sequence]
    
    # เพิ่มการตรวจสอบความต่อเนื่อง
    for i in range(1, len(cleaned_predictions)):
        if cleaned_predictions[i] != cleaned_predictions[i-1]:
            current_index = correct_sequence.index(cleaned_predictions[i])
            prev_index = correct_sequence.index(cleaned_predictions[i-1])
            if abs(current_index - prev_index) > 1:
                # ถ้าท่าทางเปลี่ยนแปลงเกินกว่า 1 ลำดับ ให้ใช้ท่าทางก่อนหน้า
                cleaned_predictions[i] = cleaned_predictions[i-1]
    
    # ตรวจสอบและเพิ่มท่าทางที่หายไป
    cleaned_predictions = ensure_all_poses(cleaned_predictions)
    
    # ตัดหรือเพิ่มข้อมูลให้มีความยาวเท่ากับข้อมูลเดิม
    if len(cleaned_predictions) > len(predictions):
        cleaned_predictions = cleaned_predictions[:len(predictions)]
    elif len(cleaned_predictions) < len(predictions):
        cleaned_predictions += [cleaned_predictions[-1]] * (len(predictions) - len(cleaned_predictions))
    
    return cleaned_predictions

def ensure_all_poses(cleaned_predictions):
    for pose in correct_sequence:
        if pose not in cleaned_predictions:
            # หาตำแหน่งที่เหมาะสมที่สุดที่จะแทรกท่าทางที่หายไป
            ideal_index = correct_sequence.index(pose)
            for i, pred in enumerate(cleaned_predictions):
                if correct_sequence.index(pred) > ideal_index:
                    cleaned_predictions.insert(i, pose)
                    break
            else:
                # ถ้าไม่มีตำแหน่งที่เหมาะสม ให้เพิ่มท่าทางที่หายไปที่ท้ายสุด
                cleaned_predictions.append(pose)
    return cleaned_predictions

# กำหนด path ของโฟลเดอร์
input_path = "./output/baseline/combined/adjusted/realtime/hpe/csv_aftermodel/predictions/feature correlation/after5fold"
output_path = "./output/baseline/combined/adjusted/realtime/hpe/csv_aftermodel/predictions/feature correlation/after5fold_cleaned_dtw"

# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์ที่ clean แล้ว
os.makedirs(output_path, exist_ok=True)

# วนลูปทุกไฟล์ใน input_path
for filename in os.listdir(input_path):
    if filename.endswith(".csv"):
        input_file = os.path.join(input_path, filename)
        output_file = os.path.join(output_path, f"cleaned_dtw_{filename}")
        
        # อ่านไฟล์ CSV
        df = pd.read_csv(input_file)
        
        # ทำความสะอาดข้อมูล
        original_predictions = df['Predicted_Pose'].tolist()
        cleaned_predictions = clean_golf_swing_data_dtw(original_predictions)
        
        # ตรวจสอบความยาวของข้อมูล
        if len(cleaned_predictions) != len(df):
            print(f"Warning: Length mismatch in {filename}. Original: {len(df)}, Cleaned: {len(cleaned_predictions)}")
        
        # อัปเดตคอลัมน์ Predicted_Pose ด้วยข้อมูลที่ clean แล้ว
        df['Predicted_Pose'] = cleaned_predictions
        
        # บันทึกไฟล์ CSV ที่ clean แล้ว
        df.to_csv(output_file, index=False)
        
        print(f"Cleaned file saved: {output_file}")

print("Data cleaning completed using DTW.")