import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# กำหนด paths
INPUT_CSV_PATH = './output/baseline/combined/adjusted/realtime/hpe/csv'
MODEL_SAVE_PATH = "./output/videos_raw/model/combined/realtime/feature correlation/epoch 50/"
OUTPUT_PATH = "./output/baseline/combined/adjusted/realtime/hpe/csv_aftermodel/predictions/feature correlation/epoch 50"

# โหลดโมเดล
model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_PATH, 'lstm_golf_swing_model.h5'))

# โหลด Label Encoder และ Scaler
le = joblib.load(os.path.join(MODEL_SAVE_PATH, 'label_encoder.joblib'))
scaler = joblib.load(os.path.join(MODEL_SAVE_PATH, 'scaler.joblib'))

# ฟังก์ชันสำหรับเตรียมข้อมูล
def prepare_data(df):
    # แยก x, y coordinates
    for joint in ['Left Wrist', 'Right Wrist', 'Left Ankle', 'Right Ankle', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
                          'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee']:
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

# ฟังก์ชันสำหรับทำนาย
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

# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# ทำนายสำหรับทุกไฟล์ CSV ในโฟลเดอร์ input
for filename in os.listdir(INPUT_CSV_PATH):
    if filename.endswith('.csv'):
        input_file_path = os.path.join(INPUT_CSV_PATH, filename)
        output_file_path = os.path.join(OUTPUT_PATH, f'predicted_{filename}')
        
        print(f"Processing file: {filename}")
        df_with_predictions = predict_poses(input_file_path)
        df_with_predictions.to_csv(output_file_path, index=False)
        print(f"Predictions saved to: {output_file_path}")

print("All predictions completed.")
