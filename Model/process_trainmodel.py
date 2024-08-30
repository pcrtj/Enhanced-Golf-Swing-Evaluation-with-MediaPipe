import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import pickle
from tqdm import tqdm

# เส้นทาง
input_folder = "./output/videos_keyframedetection/adjusted_data/other/hpe_other/spine_smoothed_csv"
model_save_path = "./output/videos_keyframedetection/adjusted_data/other/hpe_other/model"

# ฟีเจอร์
features = [
    'Left Shoulder Angle', 'Right Shoulder Angle',
    'Left Elbow Angle', 'Right Elbow Angle',
    'Left Hip Angle', 'Right Hip Angle',
    'Left Knee Angle', 'Right Knee Angle'
]

# อ่านข้อมูล
data = []
print("Reading CSV files...")
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)
        data.append(df[features])

all_data = pd.concat(data, ignore_index=True)

# เตรียมข้อมูล
X = all_data[features]
y = all_data[features]

# แบ่งข้อมูล
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ฟีเจอร์เชิงพหุนาม
degree = 2
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# รายชื่อโมเดลที่ใช้
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoostRegressor": AdaBoostRegressor(n_estimators=100, random_state=42),
    "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "HuberRegressor": HuberRegressor()
}

# ฝึกและประเมินโมเดล
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_poly, y_train)

    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test_poly)
    r2_scores = {feature: r2_score(y_test[feature], y_pred[:, i]) for i, feature in enumerate(features)}

    # บันทึกโมเดล
    model_path = os.path.join(model_save_path, f"{model_name}_model.pkl")
    print(f"Saving {model_name}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    
    print(f"Model training complete and saved as '{model_path}'")
    for feature, score in r2_scores.items():
        print(f"R² score for {feature} with {model_name}: {score:.4f}")

# Function to test a new sample and show the percentage similarity
def test_sample(sample, model):
    sample_poly = poly.transform([sample])
    pred = model.predict(sample_poly)
    similarity = 100 - abs((sample - pred[0]) / sample * 100)
    average_similarity = similarity.mean()
    return similarity, average_similarity

# Test with a sample (replace with your actual sample data)
sample_data = [45, 50, 90, 85, 120, 115, 140, 135]  # Example sample data

for model_name, model in models.items():
    similarity, average_similarity = test_sample(sample_data, model)
    print(f"{model_name} similarity percentage: {similarity}")
    print(f"{model_name} average similarity percentage: {average_similarity}")

# ฟังก์ชันทดสอบ
def test_with_csv(file_path, model):
    df = pd.read_csv(file_path)
    sample_data = df[features].values
    sample_poly = poly.transform(sample_data)
    preds = model.predict(sample_poly)
    similarity_percentages = 100 - abs((sample_data - preds) / sample_data * 100)
    average_similarity = similarity_percentages.mean(axis=0)
    return similarity_percentages, average_similarity

test_file_path = "./output/videos_keyframedetection/adjusted_data/down_the_line/hpe_down_the_line/smoothed_csv/keyframe_0_angles.csv"

for model_name, model in models.items():
    similarity_percentages, average_similarity = test_with_csv(test_file_path, model)
    print(f"{model_name} similarity percentages:\n{similarity_percentages}")
    print(f"{model_name} average similarity percentage for each feature:\n{average_similarity}")
    print(f"{model_name} overall average similarity percentage: {average_similarity.mean()}")
