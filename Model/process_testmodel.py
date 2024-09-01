import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

# กำหนดฟีเจอร์ที่ต้องการใช้
FEATURE_COLUMNS = [
    'Left Shoulder Angle', 'Right Shoulder Angle',
    'Left Elbow Angle', 'Right Elbow Angle',
    'Left Hip Angle', 'Right Hip Angle',
    'Left Knee Angle', 'Right Knee Angle'
]

# ฟังก์ชั่นสำหรับโหลดข้อมูลจาก CSV
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    X = df[FEATURE_COLUMNS].values
    y = df['Pose'].values  # คอลัมน์ 'Pose' คือ label
    return X, y

# โหลดโมเดลจากไฟล์ .pkl
model_path = './output/videos_raw/model/golf_pose_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# กำหนด path ของไฟล์ CSV เดี่ยว
csv_file_path = './output/videos_raw/csv/1046_angles.csv'

# โหลดข้อมูลและทำนาย
X_test, true_labels = load_data(csv_file_path)
predictions = model.predict(X_test)

# คำนวณความแม่นยำ
accuracy = accuracy_score(true_labels, predictions)

# แสดงผลการทำนายและความแม่นยำ
print(f"Predictions for {csv_file_path}:")
print(predictions)
print(f"Accuracy of predicting golf swing sequences: {accuracy:.2%}")
