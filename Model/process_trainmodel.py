import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, GlobalMaxPooling1D, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import joblib
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import time

CSV_FOLDER = "./output/videos_raw/csv/combined/realtime"
MODEL_SAVE_PATH = "./output/videos_raw/model/combined/realtime/feature correlation/after5fold/"

class TqdmProgressCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.progress_bar = tqdm(total=epochs, desc="Training", unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)

    def on_train_end(self, logs=None):
        self.progress_bar.close()

def load_and_prepare_data(csv_folder):
    all_data = []
    for filename in os.listdir(csv_folder):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(csv_folder, filename))
            
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

def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    
    y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    
    y_test_bin = to_categorical(y_test)
    y_pred_bin = to_categorical(y_pred_classes)
    roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average='weighted', multi_class='ovr')
    
    time_per_data = (end_time - start_time) / len(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Time per Data': time_per_data,
        'Confusion Matrix': cm
    }

def train_and_evaluate_model(model, X, y, model_name, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
            model.fit(X_train, y_train)
            fold_metrics = evaluate_model(model, X_val, y_val)
        else:  # Deep learning models
            if model_name == "1D CNN":
                X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            else:
                X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_val_reshaped = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
            y_train_cat = to_categorical(y_train)
            y_val_cat = to_categorical(y_val)
            
            model.fit(X_train_reshaped, y_train_cat, epochs=30, batch_size=32, 
                      validation_data=(X_val_reshaped, y_val_cat), verbose=0,
                      callbacks=[TqdmProgressCallback(epochs=30)])
            fold_metrics = evaluate_model(model, X_val_reshaped, y_val)
        
        fold_results.append(fold_metrics)
        print(f"Fold {fold} completed for {model_name}")
    
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_results]) 
                   for metric in fold_results[0].keys() if metric != 'Confusion Matrix'}
    
    # Sum up confusion matrices from all folds
    avg_metrics['Confusion Matrix'] = sum([fold['Confusion Matrix'] for fold in fold_results])
    
    return avg_metrics

def print_results_table(results):
    headers = ["Model"] + [metric for metric in next(iter(results.values())).keys() if metric != 'Confusion Matrix']
    table_data = [[model] + [metrics[metric] for metric in headers[1:]] for model, metrics in results.items()]
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

def plot_confusion_matrix(cm, class_names, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(MODEL_SAVE_PATH, f'confusion_matrix_{model_name}.png'))
    plt.close()

def plot_roc_curves(results, X, y):
    plt.figure(figsize=(10, 8))
    
    for model_name, metrics in results.items():
        if model_name in ['Decision Tree', 'Random Forest']:
            y_pred_proba = models[model_name].predict_proba(X)
        else:  # Deep learning models
            if model_name == "1D CNN":
                X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
            else:
                X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
            y_pred_proba = models[model_name].predict(X_reshaped)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(np.unique(y))):
            fpr[i], tpr[i], _ = roc_curve((y == i).astype(int), y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(to_categorical(y).ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'{model_name} (AUC = {roc_auc["micro"]:.4f})',
                 linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'roc_curves.png'))
    plt.close()

# Main execution
X, y = load_and_prepare_data(CSV_FOLDER)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = {}
models = {}

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
results['Decision Tree'] = train_and_evaluate_model(dt_model, X_scaled, y_encoded, "Decision Tree")
models['Decision Tree'] = dt_model

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
results['Random Forest'] = train_and_evaluate_model(rf_model, X_scaled, y_encoded, "Random Forest")
models['Random Forest'] = rf_model

# LSTM
lstm_model = Sequential([
    Input(shape=(1, X_scaled.shape[1])),
    LSTM(100, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
results['LSTM'] = train_and_evaluate_model(lstm_model, X_scaled, y_encoded, "LSTM")
models['LSTM'] = lstm_model

# GRU
gru_model = Sequential([
    Input(shape=(1, X_scaled.shape[1])),
    GRU(100, return_sequences=True),
    Dropout(0.2),
    GRU(50),
    Dropout(0.2),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])
gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
results['GRU'] = train_and_evaluate_model(gru_model, X_scaled, y_encoded, "GRU")
models['GRU'] = gru_model

# 1D CNN
cnn_model = Sequential([
    Input(shape=(X_scaled.shape[1], 1)),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    GlobalMaxPooling1D(),
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
results['1D CNN'] = train_and_evaluate_model(cnn_model, X_scaled, y_encoded, "1D CNN")
models['1D CNN'] = cnn_model

# Print results table
print("\nModel Performance Summary (5-Fold Cross-Validation):")
print_results_table(results)

# Plot ROC curves
plot_roc_curves(results, X_scaled, y_encoded)

# Plot confusion matrices
for model_name, model_metrics in results.items():
    cm = model_metrics['Confusion Matrix']
    plot_confusion_matrix(cm, le.classes_, model_name)

# Save models and encoders
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

joblib.dump(dt_model, os.path.join(MODEL_SAVE_PATH, 'decision_tree_model.joblib'))
joblib.dump(rf_model, os.path.join(MODEL_SAVE_PATH, 'random_forest_model.joblib'))
lstm_model.save(os.path.join(MODEL_SAVE_PATH, 'lstm_model.h5'))
gru_model.save(os.path.join(MODEL_SAVE_PATH, 'gru_model.h5'))
cnn_model.save(os.path.join(MODEL_SAVE_PATH, 'cnn_model.h5'))
joblib.dump(le, os.path.join(MODEL_SAVE_PATH, 'label_encoder.joblib'))
joblib.dump(scaler, os.path.join(MODEL_SAVE_PATH, 'scaler.joblib'))

print("\nAll models have been trained and saved successfully.")
print("ROC curves have been plotted and saved as 'roc_curves.png'.")
print("Confusion matrices have been plotted and saved for each model.")


# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.utils import to_categorical

# # กำหนด path
# CSV_FOLDER = "./output/videos_raw/csv/combined/realtime"
# MODEL_SAVE_PATH = "./output/videos_raw/model/combined/realtime/feature correlation/epoch 30"

# # ฟังก์ชันสำหรับโหลดและเตรียมข้อมูล
# def load_and_prepare_data(csv_folder):
#     all_data = []
#     for filename in os.listdir(csv_folder):
#         if filename.endswith('.csv'):
#             df = pd.read_csv(os.path.join(csv_folder, filename))
            
#             # แยก x, y coordinates
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

# # โหลดและเตรียมข้อมูล
# X, y = load_and_prepare_data(CSV_FOLDER)

# # แปลง labels เป็นตัวเลข
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)

# # แบ่งข้อมูลเป็น training และ testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# # Normalize ข้อมูล
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Reshape ข้อมูลสำหรับ LSTM (samples, time steps, features)
# X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
# X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# # แปลง labels เป็น one-hot encoding
# y_train_cat = to_categorical(y_train)
# y_test_cat = to_categorical(y_test)

# # สร้างโมเดล LSTM
# model = Sequential([
#     LSTM(100, input_shape=(1, X_train_reshaped.shape[2]), return_sequences=True),
#     Dropout(0.2),
#     LSTM(50),
#     Dropout(0.2),
#     Dense(len(le.classes_), activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # เทรนโมเดล
# history = model.fit(X_train_reshaped, y_train_cat, epochs=30, batch_size=32, 
#                     validation_data=(X_test_reshaped, y_test_cat), verbose=1)

# # บันทึกโมเดล
# if not os.path.exists(MODEL_SAVE_PATH):
#     os.makedirs(MODEL_SAVE_PATH)
# model.save(os.path.join(MODEL_SAVE_PATH, 'lstm_golf_swing_model.h5'))

# # ประเมินโมเดล
# test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
# print(f"Test accuracy: {test_accuracy:.4f}")

# # บันทึก Label Encoder
# import joblib
# joblib.dump(le, os.path.join(MODEL_SAVE_PATH, 'label_encoder.joblib'))

# # บันทึก Scaler
# joblib.dump(scaler, os.path.join(MODEL_SAVE_PATH, 'scaler.joblib'))
