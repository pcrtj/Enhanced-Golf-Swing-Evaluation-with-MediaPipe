import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# Constants
CSV_SAMPLE_PATH = "./output/videos_raw/csv/face_on/1005_angles.csv"  # Path to your sample CSV file
MODEL_SAVE_PATH = "./output/videos_raw/model/face_on"

# Function to load data from the sample CSV file
def load_sample_data(csv_file):
    print("Loading sample data...")
    df = pd.read_csv(csv_file)
    X_sample = df[['Left Shoulder Angle', 'Right Shoulder Angle', 'Left Elbow Angle', 
                   'Right Elbow Angle', 'Left Hip Angle', 'Right Hip Angle', 
                   'Left Knee Angle', 'Right Knee Angle']]
    y_sample = df['Pose']
    print("Sample data loaded.")
    return X_sample, y_sample

# Function to load the LabelEncoder used during training
def load_label_encoder(model_path):
    with open(os.path.join(model_path, "label_encoder.pkl"), 'rb') as f:
        le = pickle.load(f)
    return le

# Function to load and test each model
def test_models(X_sample, y_sample):
    # Load the LabelEncoder
    le = load_label_encoder(MODEL_SAVE_PATH)
    y_encoded = le.transform(y_sample)
    
    # Reshape input for Keras models
    X_sample_reshaped = X_sample.values.reshape((X_sample.shape[0], X_sample.shape[1], 1))
    
    results = {}  # Dictionary to store accuracy of each model
    
    # Test RandomForestClassifier
    print("Testing RandomForestClassifier...")
    with open(os.path.join(MODEL_SAVE_PATH, "random_forest_classifier.pkl"), 'rb') as f:
        rf_clf = pickle.load(f)
    y_pred_rf = rf_clf.predict(X_sample)
    accuracy_rf = accuracy_score(y_encoded, y_pred_rf)
    results['RandomForestClassifier'] = accuracy_rf
    print(f"RandomForestClassifier Accuracy: {accuracy_rf:.2f}")
    
    # Test LSTM Model
    print("Testing LSTM model...")
    lstm_model = load_model(os.path.join(MODEL_SAVE_PATH, "lstm_model.h5"))
    y_pred_lstm = lstm_model.predict(X_sample_reshaped).argmax(axis=1)
    accuracy_lstm = accuracy_score(y_encoded, y_pred_lstm)
    results['LSTM'] = accuracy_lstm
    print(f"LSTM Accuracy: {accuracy_lstm:.2f}")
    
    # Test GRU Model
    print("Testing GRU model...")
    gru_model = load_model(os.path.join(MODEL_SAVE_PATH, "gru_model.h5"))
    y_pred_gru = gru_model.predict(X_sample_reshaped).argmax(axis=1)
    accuracy_gru = accuracy_score(y_encoded, y_pred_gru)
    results['GRU'] = accuracy_gru
    print(f"GRU Accuracy: {accuracy_gru:.2f}")
    
    # Test 1D CNN Model
    print("Testing 1D CNN model...")
    cnn_model = load_model(os.path.join(MODEL_SAVE_PATH, "cnn_model.h5"))
    y_pred_cnn = cnn_model.predict(X_sample_reshaped).argmax(axis=1)
    accuracy_cnn = accuracy_score(y_encoded, y_pred_cnn)
    results['1D CNN'] = accuracy_cnn
    print(f"1D CNN Accuracy: {accuracy_cnn:.2f}")
    
    # Print summary of results
    print("\nTesting Summary:")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.2f}")

# Main execution
if __name__ == "__main__":
    X_sample, y_sample = load_sample_data(CSV_SAMPLE_PATH)
    test_models(X_sample, y_sample)
    print("Testing process complete.")
