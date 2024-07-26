import pandas as pd
import os


input_folder = './output/videos_keyframedetection/raw_data/hpe_raw_data/spine_smooth_csv'
output_folder = './output/videos_keyframedetection/raw_data/hpe_raw_data/spine_smooth_csv'

columns_to_split = [
    'x, y Left Shoulder', 'x, y Right Shoulder', 'x, y Left Elbow', 'x, y Right Elbow',
    'x, y Left Hip', 'x, y Right Hip', 'x, y Left Knee', 'x, y Right Knee',
    'x, y Left Wrist', 'x, y Right Wrist', 'x, y Left Ankle', 'x, y Right Ankle', 'x, y Nose'
]

for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        input_csv = os.path.join(input_folder, filename)
        output_csv = os.path.join(output_folder, f'separated_{filename}')

        df = pd.read_csv(input_csv)

        for col in columns_to_split:
            new_x_col = col.replace('x, y', 'x').replace('X, Y', 'X').strip()
            new_y_col = col.replace('x, y', 'y').replace('X, Y', 'Y').strip()
            if col in df.columns:
                df[[new_x_col, new_y_col]] = df[col].str.split(',', expand=True).astype(float)
                df.drop(columns=[col], inplace=True)

        df.to_csv(output_csv, index=False)
        print(f'ไฟล์ถูกบันทึกที่ {output_csv}')
