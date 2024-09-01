import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
import pickle

# Constants
CSV_FOLDER = "./output/videos_raw/csv/face_on"
MODEL_SAVE_PATH = "./output/videos_raw/model/face_on"

# Function to load and combine data from CSV files
def load_and_prepare_data(csv_folder):
    print("Loading data from CSV files...")
    all_data = []
    for filename in os.listdir(csv_folder):
        if filename.endswith('_angles.csv'):
            df = pd.read_csv(os.path.join(csv_folder, filename))
            all_data.append(df)
            print(f"Loaded {filename}")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print("Data loading complete.")
    
    # Selecting features and target
    X = combined_data[['Left Shoulder Angle', 'Right Shoulder Angle', 'Left Elbow Angle', 
                       'Right Elbow Angle', 'Left Hip Angle', 'Right Hip Angle', 
                       'Left Knee Angle', 'Right Knee Angle']]
    y = combined_data['Pose']
    
    return X, y

# Function to train and save models
def train_and_save_models(X, y):
    # Ensure the directory exists
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # บันทึก LabelEncoder
    with open(os.path.join(MODEL_SAVE_PATH, "label_encoder.pkl"), 'wb') as f:
        pickle.dump(le, f)
    print("LabelEncoder saved.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    results = {}  # Dictionary to store accuracy and MSE of each model

    # RandomForestClassifier
    print("Training RandomForestClassifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results['RandomForestClassifier'] = accuracy
    print(f"RandomForestClassifier Accuracy: {accuracy:.2f}")
    with open(os.path.join(MODEL_SAVE_PATH, "random_forest_classifier.pkl"), 'wb') as f:
        pickle.dump(rf_clf, f)
    print("RandomForestClassifier model saved.")
    
    # RandomForestRegressor
    print("Training RandomForestRegressor...")
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)  # Use encoded labels
    y_pred_reg = rf_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_reg)
    results['RandomForestRegressor MSE'] = mse
    print(f"RandomForestRegressor MSE: {mse:.2f}")
    with open(os.path.join(MODEL_SAVE_PATH, "random_forest_regressor.pkl"), 'wb') as f:
        pickle.dump(rf_reg, f)
    print("RandomForestRegressor model saved.")
    
    # LSTM Model
    print("Training LSTM model...")
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_model = Sequential([
        LSTM(64, input_shape=(X_train_reshaped.shape[1], 1), activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])
    lstm_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2)
    loss, lstm_accuracy = lstm_model.evaluate(X_test_reshaped, y_test)
    results['LSTM'] = lstm_accuracy
    print(f"LSTM Accuracy: {lstm_accuracy:.2f}")
    lstm_model.save(os.path.join(MODEL_SAVE_PATH, "lstm_model.h5"))
    print("LSTM model saved.")
    
    # GRU Model
    print("Training GRU model...")
    gru_model = Sequential([
        GRU(64, input_shape=(X_train_reshaped.shape[1], 1), activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])
    gru_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    gru_model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2)
    loss, gru_accuracy = gru_model.evaluate(X_test_reshaped, y_test)
    results['GRU'] = gru_accuracy
    print(f"GRU Accuracy: {gru_accuracy:.2f}")
    gru_model.save(os.path.join(MODEL_SAVE_PATH, "gru_model.h5"))
    print("GRU model saved.")
    
    # 1D CNN Model
    print("Training 1D CNN model...")
    cnn_model = Sequential([
        Conv1D(64, 2, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
        MaxPooling1D(2),
        Flatten(),
        Dense(len(le.classes_), activation='softmax')
    ])
    cnn_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2)
    loss, cnn_accuracy = cnn_model.evaluate(X_test_reshaped, y_test)
    results['1D CNN'] = cnn_accuracy
    print(f"1D CNN Accuracy: {cnn_accuracy:.2f}")
    cnn_model.save(os.path.join(MODEL_SAVE_PATH, "cnn_model.h5"))
    print("1D CNN model saved.")
    
    # Print summary of results
    print("\nTraining Summary:")
    for model_name, metric in results.items():
        print(f"{model_name}: {metric:.2f}")

# Main execution
if __name__ == "__main__":
    X, y = load_and_prepare_data(CSV_FOLDER)
    train_and_save_models(X, y)
    print("Training process complete.")
