import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    angle_columns = ['Left Shoulder Angle', 'Right Shoulder Angle', 'Left Elbow Angle', 'Right Elbow Angle',
                     'Left Hip Angle', 'Right Hip Angle', 'Left Knee Angle', 'Right Knee Angle']
    return df[angle_columns + ['Predicted_Pose']]

baseline_data = []
baseline_folder = './output/baseline/combined/adjusted/realtime/hpe/csv_aftermodel/predictions/feature correlation/epoch 50'
for file in os.listdir(baseline_folder):
    if file.endswith('.csv'):
        df = load_data(os.path.join(baseline_folder, file))
        baseline_data.append(df)

user_data = load_data('./output/baseline/combined/adjusted/realtime/hpe/csv_aftermodel/predictions/feature correlation/epoch 50/predicted_0.csv')

poses = ['Address', 'Toe-Up', 'Mid-Backswing', 'Top', 'Mid-Downswing', 'Impact', 'Mid-Follow-Through', 'Finish']

def split_data_by_pose(data):
    return {pose: data[data['Predicted_Pose'] == pose].drop('Predicted_Pose', axis=1) for pose in poses}

baseline_poses = split_data_by_pose(pd.concat(baseline_data))
user_poses = split_data_by_pose(user_data)

def calculate_similarity(baseline, user):
    if baseline.empty or user.empty:
        return None
    return cosine_similarity(baseline.mean().values.reshape(1, -1), user.mean().values.reshape(1, -1))[0][0]

similarities = {}
for pose in poses:
    if pose in baseline_poses and pose in user_poses:
        similarity = calculate_similarity(baseline_poses[pose], user_poses[pose])
        if similarity is not None:
            similarities[pose] = similarity

if similarities:
    average_similarity = np.mean(list(similarities.values()))
else:
    average_similarity = None

print("Cosine Similarity for each pose (%):")
for pose, similarity in similarities.items():
    print(f"{pose}: {similarity * 100:.2f}%")

if average_similarity is not None:
    print(f"\nAverage Similarity: {average_similarity * 100:.2f}%")
else:
    print("\nNo valid similarities found.")