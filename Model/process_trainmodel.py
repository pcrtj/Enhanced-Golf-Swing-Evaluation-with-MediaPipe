# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import Callback
# import joblib
# from tqdm import tqdm

# CSV_FOLDER = "./output/videos_raw/csv/combined/realtime"
# MODEL_SAVE_PATH = "./output/videos_raw/model/combined/realtime/feature correlation/compare model"

# class TqdmProgressCallback(Callback):
#     def __init__(self, epochs):
#         super().__init__()
#         self.progress_bar = tqdm(total=epochs, desc="Training", unit="epoch")

#     def on_epoch_end(self, epoch, logs=None):
#         self.progress_bar.update(1)

#     def on_train_end(self, logs=None):
#         self.progress_bar.close()

# def load_and_prepare_data(csv_folder):
#     all_data = []
#     for filename in os.listdir(csv_folder):
#         if filename.endswith('.csv'):
#             df = pd.read_csv(os.path.join(csv_folder, filename))
            
#             for joint in ['Left Wrist', 'Right Wrist', 'Left Ankle', 'Right Ankle', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
#                           'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee']:
#                 df[[f'{joint} x', f'{joint} y']] = df[f'x, y {joint}'].str.split(', ', expand=True).astype(float)
            
#             all_data.append(df)

#     combined_data = pd.concat(all_data, ignore_index=True)
    
#     feature_columns = [
#     'Time',
#     'Left Shoulder Angle', 'Right Shoulder Angle',
#     'Left Elbow Angle', 'Right Elbow Angle',
#     'Left Hip Angle', 'Right Hip Angle',
#     'Left Knee Angle', 'Right Knee Angle',
#     'Left Shoulder x', 'Left Shoulder y',
#     'Right Shoulder x', 'Right Shoulder y',
#     'Left Elbow x', 'Left Elbow y',
#     'Right Elbow x', 'Right Elbow y',
#     'Left Hip x', 'Left Hip y',
#     'Right Hip x', 'Right Hip y',
#     'Left Knee x', 'Left Knee y',
#     'Right Knee x', 'Right Knee y',
#     'Left Wrist x', 'Left Wrist y',
#     'Right Wrist x', 'Right Wrist y',
#     'Left Ankle x', 'Left Ankle y',
#     'Right Ankle x', 'Right Ankle y'
#     ]
    
#     X = combined_data[feature_columns].values
#     y = combined_data['Pose'].values
    
#     return X, y

# X, y = load_and_prepare_data(CSV_FOLDER)

# le = LabelEncoder()
# y_encoded = le.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# results = {}

# print("Training Decision Tree model...")
# dt_model = DecisionTreeClassifier(random_state=42)
# dt_model.fit(X_train_scaled, y_train)
# dt_accuracy = dt_model.score(X_test_scaled, y_test)
# results['Decision Tree'] = dt_accuracy

# print("Training Random Forest model...")
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train_scaled, y_train)
# rf_accuracy = rf_model.score(X_test_scaled, y_test)
# results['Random Forest'] = rf_accuracy

# X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
# X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
# y_train_cat = to_categorical(y_train)
# y_test_cat = to_categorical(y_test)

# def train_dl_model(model, model_name, epochs=50, batch_size=32):
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     print(f"Training {model_name} model...")
#     tqdm_callback = TqdmProgressCallback(epochs)
#     history = model.fit(X_train_reshaped, y_train_cat, epochs=epochs, batch_size=batch_size, 
#                         validation_data=(X_test_reshaped, y_test_cat), verbose=0,
#                         callbacks=[tqdm_callback])
#     test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
#     results[model_name] = test_accuracy
#     return model, history

# lstm_model = Sequential([
#     LSTM(100, input_shape=(1, X_train_reshaped.shape[2]), return_sequences=True),
#     Dropout(0.2),
#     LSTM(50),
#     Dropout(0.2),
#     Dense(len(le.classes_), activation='softmax')
# ])
# lstm_model, lstm_history = train_dl_model(lstm_model, "LSTM")

# gru_model = Sequential([
#     GRU(100, input_shape=(1, X_train_reshaped.shape[2]), return_sequences=True),
#     Dropout(0.2),
#     GRU(50),
#     Dropout(0.2),
#     Dense(len(le.classes_), activation='softmax')
# ])
# gru_model, gru_history = train_dl_model(gru_model, "GRU")

# cnn_model = Sequential([
#     Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, X_train_reshaped.shape[2])),
#     Flatten(),
#     Dense(100, activation='relu'),
#     Dropout(0.2),
#     Dense(len(le.classes_), activation='softmax')
# ])
# cnn_model, cnn_history = train_dl_model(cnn_model, "1D CNN")

# if not os.path.exists(MODEL_SAVE_PATH):
#     os.makedirs(MODEL_SAVE_PATH)

# joblib.dump(dt_model, os.path.join(MODEL_SAVE_PATH, 'decision_tree_model.joblib'))
# joblib.dump(rf_model, os.path.join(MODEL_SAVE_PATH, 'random_forest_model.joblib'))
# lstm_model.save(os.path.join(MODEL_SAVE_PATH, 'lstm_model.h5'))
# gru_model.save(os.path.join(MODEL_SAVE_PATH, 'gru_model.h5'))
# cnn_model.save(os.path.join(MODEL_SAVE_PATH, 'cnn_model.h5'))
# joblib.dump(le, os.path.join(MODEL_SAVE_PATH, 'label_encoder.joblib'))
# joblib.dump(scaler, os.path.join(MODEL_SAVE_PATH, 'scaler.joblib'))

# print("\nAll models have been trained and saved successfully.")

# print("\nModel Accuracy Summary:")
# for model, accuracy in results.items():
#     print(f"{model}: {accuracy:.4f}")


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# กำหนด path
CSV_FOLDER = "./output/videos_raw/csv/combined/realtime"
MODEL_SAVE_PATH = "./output/videos_raw/model/combined/realtime/feature correlation/epoch 30"

# ฟังก์ชันสำหรับโหลดและเตรียมข้อมูล
def load_and_prepare_data(csv_folder):
    all_data = []
    for filename in os.listdir(csv_folder):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(csv_folder, filename))
            
            # แยก x, y coordinates
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
    
    return X, y

# โหลดและเตรียมข้อมูล
X, y = load_and_prepare_data(CSV_FOLDER)

# แปลง labels เป็นตัวเลข
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# แบ่งข้อมูลเป็น training และ testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalize ข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape ข้อมูลสำหรับ LSTM (samples, time steps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# แปลง labels เป็น one-hot encoding
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# สร้างโมเดล LSTM
model = Sequential([
    LSTM(100, input_shape=(1, X_train_reshaped.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# เทรนโมเดล
history = model.fit(X_train_reshaped, y_train_cat, epochs=30, batch_size=32, 
                    validation_data=(X_test_reshaped, y_test_cat), verbose=1)

# บันทึกโมเดล
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
model.save(os.path.join(MODEL_SAVE_PATH, 'lstm_golf_swing_model.h5'))

# ประเมินโมเดล
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# บันทึก Label Encoder
import joblib
joblib.dump(le, os.path.join(MODEL_SAVE_PATH, 'label_encoder.joblib'))

# บันทึก Scaler
joblib.dump(scaler, os.path.join(MODEL_SAVE_PATH, 'scaler.joblib'))
