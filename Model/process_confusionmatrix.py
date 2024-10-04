import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from tensorflow.keras.models import load_model

# กำหนด path
MODEL_SAVE_PATH = "./output/videos_raw/model/combined/realtime/feature correlation/after5fold/"
DATA_PATH = "./output/videos_raw/csv/combined/realtime"

def load_data():
    all_data = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(DATA_PATH, filename))
            
            for joint in ['Left Wrist', 'Right Wrist', 'Left Ankle', 'Right Ankle', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
                          'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee']:
                df[[f'{joint} x', f'{joint} y']] = df[f'x, y {joint}'].str.split(', ', expand=True).astype(float)
            
            all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True)
    
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
    
    X = combined_data[feature_columns].values
    y = combined_data['Pose'].values
    
    # แบ่งข้อมูลเป็นชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize ข้อมูล
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    return X_test_scaled, y_test

def plot_confusion_matrix(cm, class_names, model_name):
    # Define the mapping from full names to abbreviations
    name_to_abbr = {
        'Address': 'A',
        'Toe-Up': 'TU',
        'Mid-Backswing': 'MB',
        'Top': 'T',
        'Mid-Downswing': 'MD',
        'Impact': 'I',
        'Mid-Follow-Through': 'MFT',
        'Finish': 'F',
        'Preparation': 'P'  # ยังคงมี Preparation แต่จะไม่แสดงในการ plot
    }

    # Define the correct order of events (ไม่รวม Preparation)
    correct_order = ['A', 'TU', 'MB', 'T', 'MD', 'I', 'MFT', 'F']
    
    # Create a mapping from the original index to the new index
    index_mapping = {name: i for i, name in enumerate(class_names) if name != 'Preparation'}
    
    # Reorder the confusion matrix
    cm_ordered = np.zeros((len(correct_order), len(correct_order)))
    for i, row in enumerate(class_names):
        if row == 'Preparation':
            continue
        for j, col in enumerate(class_names):
            if col == 'Preparation':
                continue
            abbr_row = name_to_abbr.get(row, row)
            abbr_col = name_to_abbr.get(col, col)
            if abbr_row in correct_order and abbr_col in correct_order:
                new_i = correct_order.index(abbr_row)
                new_j = correct_order.index(abbr_col)
                cm_ordered[new_i, new_j] = cm[i, j]
    
    # Convert to percentages
    row_sums = cm_ordered.sum(axis=1, keepdims=True)
    cm_percent = np.where(row_sums != 0, (cm_ordered / row_sums * 100).round(2), 0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=correct_order,
                yticklabels=correct_order,
                vmin=0, vmax=100)
    plt.title(f'Confusion Matrix (%) - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, f'confusion_matrix_{model_name}.png'))
    plt.close()

def main():
    # โหลด Label Encoder
    le = joblib.load(os.path.join(MODEL_SAVE_PATH, 'label_encoder.joblib'))
    
    # โหลดข้อมูลทดสอบ
    X_test, y_test = load_data()
    
    # แปลง y_test เป็นตัวเลข
    y_test_encoded = le.transform(y_test)
    
    # โหลดโมเดลและสร้าง confusion matrix
    models = {
        'Decision Tree': joblib.load(os.path.join(MODEL_SAVE_PATH, 'decision_tree_model.joblib')),
        'Random Forest': joblib.load(os.path.join(MODEL_SAVE_PATH, 'random_forest_model.joblib')),
        'LSTM': load_model(os.path.join(MODEL_SAVE_PATH, 'lstm_model.h5')),
        'GRU': load_model(os.path.join(MODEL_SAVE_PATH, 'gru_model.h5')),
        '1D CNN': load_model(os.path.join(MODEL_SAVE_PATH, 'cnn_model.h5'))
    }
    
    for model_name, model in models.items():
        if model_name in ['Decision Tree', 'Random Forest']:
            y_pred = model.predict(X_test)
        else:  # Deep learning models
            if model_name == '1D CNN':
                X_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            else:
                X_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            y_pred = np.argmax(model.predict(X_reshaped), axis=1)
        
        cm = confusion_matrix(y_test_encoded, y_pred)
        plot_confusion_matrix(cm, le.classes_, model_name)
    
    print("All confusion matrices have been plotted and saved.")

if __name__ == "__main__":
    main()