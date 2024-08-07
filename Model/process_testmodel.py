import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# Paths to the CSV file
CSV_PATH = "./output/videos_keyframedetection/adjusted_data/down_the_line/hpe_down_the_line/spine_smoothed_csv/spine_keyframe_1026_angles.csv"

# Default feature columns
FEATURE_COLUMNS = [
    'Left Shoulder Angle', 'Right Shoulder Angle',
    'Left Elbow Angle', 'Right Elbow Angle',
    'Left Hip Angle', 'Right Hip Angle',
    'Left Knee Angle', 'Right Knee Angle'
]

# Models to be tested
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5)
}

def test_with_csv(model, file_path, features, poly):
    df = pd.read_csv(file_path)
    sample_data = df[features].values
    poly.fit(sample_data)  # Fit the PolynomialFeatures instance
    sample_poly = poly.transform(sample_data)
    preds = model.predict(sample_poly)
    similarity_percentages = 100 - abs((sample_data - preds) / sample_data * 100)
    average_similarity = similarity_percentages.mean(axis=0)
    return similarity_percentages, average_similarity

def main():
    poly = PolynomialFeatures(degree=2)  # Adjust the degree as needed
    best_model = None
    best_score = -float('inf')
    
    all_scores = {}

    for model_name, model in models.items():
        print(f"Loading {model_name}...")
        model_path = f"./output/videos_keyframedetection/adjusted_data/down_the_line/hpe_down_the_line/model/{model_name}_model.pkl"
        model = joblib.load(model_path)
        
        print(f"Testing {model_name}...")
        similarity_percentages, average_similarity = test_with_csv(model, CSV_PATH, FEATURE_COLUMNS, poly)
        overall_average_similarity = average_similarity.mean()
        
        print(f"{model_name} similarity percentages:\n{similarity_percentages}")
        print(f"{model_name} average similarity percentage for each feature:\n{average_similarity}")
        print(f"{model_name} overall average similarity percentage: {overall_average_similarity}")

        all_scores[model_name] = overall_average_similarity

        if overall_average_similarity > best_score:
            best_score = overall_average_similarity
            best_model = model_name

    print("\n\n*** Model accuracy comparison ***")
    for model_name, score in all_scores.items():
        print(f"{model_name}: Overall average similarity percentage = {score:.4f} %")
    print(f"\n*** {best_model} is the most accurate model with an overall average similarity percentage of {best_score} % ***")

if __name__ == "__main__":
    main()
