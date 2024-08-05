import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import pickle
from tqdm import tqdm

# Path to the folder containing CSV files
input_folder = "./output/videos_keyframedetection/adjusted_data/down_the_line/hpe_down_the_line/spine_smoothed_csv"
# Path to save the model
model_save_path = "./output/videos_keyframedetection/adjusted_data/down_the_line/hpe_down_the_line/model/polynomial_model.pkl"

# List of features to use (excluding 'Time')
features = [
    'Left Shoulder Angle', 'Right Shoulder Angle',
    'Left Elbow Angle', 'Right Elbow Angle',
    'Left Hip Angle', 'Right Hip Angle',
    'Left Knee Angle', 'Right Knee Angle'
]

# List to store data
data = []

# Read each CSV file in the input folder
print("Reading CSV files...")
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)
        data.append(df[features])

# Concatenate all dataframes into a single dataframe
all_data = pd.concat(data, ignore_index=True)

# Define the input variables (X) and the target variables (y)
X = all_data[features]
y = all_data[features]  # Assuming you want to predict all features

# Split the data into training and testing sets (70% train, 30% test)
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Transform the features into polynomial features
degree = 2  # You can change the degree of the polynomial as needed
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Create and train the Decision Tree Regression model
model = DecisionTreeRegressor()

print("Training the model...")
for _ in tqdm(range(1)):  # For progress bar simulation
    model.fit(X_train_poly, y_train)

# Predict the target variables for the test set
y_pred = model.predict(X_test_poly)

# Calculate the R² score (coefficient of determination) for each output variable
r2_scores = {feature: r2_score(y_test[feature], y_pred[:, i]) for i, feature in enumerate(features)}

# Save the trained model to a file
print("Saving the model...")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
with open(model_save_path, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f"Model training complete and saved as '{model_save_path}'")
for feature, score in r2_scores.items():
    print(f"R² score for {feature}: {score:.4f}")

# Commenting out the test_sample function
# # Function to test a new sample and show the percentage similarity
# def test_sample(sample):
#     sample_poly = poly.transform([sample])
#     pred = model.predict(sample_poly)
#     similarity = 100 - abs((sample - pred[0]) / sample * 100)
#     average_similarity = similarity.mean()
#     return similarity, average_similarity

# # Test with a sample (replace with your actual sample data)
# sample_data = [45, 50, 90, 85, 120, 115, 140, 135]  # Example sample data
# similarity, average_similarity = test_sample(sample_data)

# print(f"Sample data: {sample_data}")
# print(f"Similarity percentage: {similarity}")
# print(f"Average similarity percentage: {average_similarity}")

# Function to test with a CSV file and show the percentage similarity
def test_with_csv(file_path):
    df = pd.read_csv(file_path)
    sample_data = df[features].values
    sample_poly = poly.transform(sample_data)
    preds = model.predict(sample_poly)
    similarity_percentages = 100 - abs((sample_data - preds) / sample_data * 100)
    average_similarity = similarity_percentages.mean(axis=0)
    return similarity_percentages, average_similarity

test_file_path = "./output/videos_keyframedetection/adjusted_data/down_the_line/hpe_down_the_line/smoothed_csv/keyframe_0_angles.csv"
similarity_percentages, average_similarity = test_with_csv(test_file_path)

print(f"Similarity percentages:\n{similarity_percentages}")
print(f"Average similarity percentage for each feature:\n{average_similarity}")
print(f"Overall average similarity percentage: {average_similarity.mean()}")
