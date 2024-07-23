import pandas as pd
import numpy as np
import os

input_path = './output/videos_keyframedetection/raw_data/hpe_raw_data/csv'
output_path = './output/videos_keyframedetection/raw_data/hpe_raw_data/spine_raw_csv'
if not os.path.exists(output_path):
    os.makedirs(output_path)

total_files = len([f for f in os.listdir(input_path) if f.endswith(".csv")])
file_counter = 0

for filename in os.listdir(input_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_path, filename)
        file_counter += 1
        print(f"Processing file {file_counter} of {total_files}: {filename}")

        data = pd.read_csv(file_path)

        data[['Left Shoulder X', 'Left Shoulder Y']] = data['x, y Left Shoulder'].str.split(',', expand=True).astype(float)
        data[['Right Shoulder X', 'Right Shoulder Y']] = data['x, y Right Shoulder'].str.split(',', expand=True).astype(float)
        data[['Left Hip X', 'Left Hip Y']] = data['x, y Left Hip'].str.split(',', expand=True).astype(float)
        data[['Right Hip X', 'Right Hip Y']] = data['x, y Right Hip'].str.split(',', expand=True).astype(float)

        data['Spine Shoulder X'] = (data['Left Shoulder X'] + data['Right Shoulder X']) / 2
        data['Spine Shoulder Y'] = (data['Left Shoulder Y'] + data['Right Shoulder Y']) / 2
        data['Spine Hip X'] = (data['Left Hip X'] + data['Right Hip X']) / 2
        data['Spine Hip Y'] = (data['Left Hip Y'] + data['Right Hip Y']) / 2

        # คำนวณมุมการโค้งข้างของกระดูกสันหลัง (Lateral Bending Angle)
        data['Lateral Bending Angle'] = np.degrees(np.arctan2(data['Spine Hip X'] - data['Spine Shoulder X'], data['Spine Hip Y'] - data['Spine Shoulder Y']))

        # แสดงทิศทางการโน้ม (Leaning) มากกว่า 0 คือโน้มไปข้างหลัง เเละ น้อยกว่า 0 คือโน้มไปข้างหน้า
        leaning = []
        for angle in data['Lateral Bending Angle']:
            if angle > 0:
                leaning.append('Trailing Side')
            else:
                leaning.append('Leading Side')
        data['Leaning'] = leaning


        data.drop(columns=['Left Shoulder X', 'Left Shoulder Y', 'Right Shoulder X', 'Right Shoulder Y',
                           'Left Hip X', 'Left Hip Y', 'Right Hip X', 'Right Hip Y', 'Spine Shoulder X',
                           'Spine Shoulder Y', 'Spine Hip X', 'Spine Hip Y'], inplace=True)

        output_filename = f"spine_{filename}"
        output_file_path = os.path.join(output_path, output_filename)
        data.to_csv(output_file_path, index=False)
