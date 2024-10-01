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