import os
import pandas as pd
import numpy as np
from dtaidistance import dtw
from scipy.spatial.distance import euclidean

FACE_ON_BASELINE_FOLDER = './output/baseline/combined/adjusted/realtime/hpe/csv_aftermodel/predictions/feature correlation/epoch 30/face_on'
DOWN_THE_LINE_BASELINE_FOLDER = './output/baseline/combined/adjusted/realtime/hpe/csv_aftermodel/predictions/feature correlation/epoch 30/down_the_line'
USER_CSV_PATH = '../Web/Backend/uploads/47/predicted_data.csv'

angle_columns = [
    'Left Shoulder Angle', 'Right Shoulder Angle', 'Left Elbow Angle', 'Right Elbow Angle',
    'Left Hip Angle', 'Right Hip Angle', 'Left Knee Angle', 'Right Knee Angle'
]

golf_pose_order = [
    'Address', 'Toe-Up', 'Mid-Backswing', 'Top', 'Mid-Downswing', 'Impact', 'Mid-Follow-Through', 'Finish'
]

def load_data(file_path):
    df = pd.read_csv(file_path)
    for column in angle_columns:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: float(str(x).split(',')[0]))
    return df

def split_data_by_pose(df):
    return {pose: group for pose, group in df.groupby('Predicted_Pose')}

def calculate_similarity(baseline, user):
    if baseline.empty or user.empty:
        return None, 0
    
    baseline_data = baseline[angle_columns].values
    user_data = user[angle_columns].values

    def safe_normalize(data):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        return (data - min_vals) / range_vals

    baseline_norm = safe_normalize(baseline_data)
    user_norm = safe_normalize(user_data)

    distances = []
    for b, u in zip(baseline_norm, user_norm):
        valid_indices = np.isfinite(b) & np.isfinite(u)
        if np.any(valid_indices):
            distances.append(euclidean(b[valid_indices], u[valid_indices]))
        else:
            distances.append(0)

    max_possible_distance = np.sqrt(len(angle_columns))  # Maximum possible distance in n-dimensional space
    similarities = [1 - (d / max_possible_distance) for d in distances]

    angle_similarities = {}
    for i, column in enumerate(angle_columns):
        angle_similarities[column] = np.mean([1 - abs(b[i] - u[i]) for b, u in zip(baseline_norm, user_norm) if np.isfinite(b[i]) and np.isfinite(u[i])])

    overall_similarity = np.mean(similarities)

    return angle_similarities, overall_similarity

def assess_similarity(baseline_folder, user_csv_path):
    baseline_data = []
    for file in os.listdir(baseline_folder):
        if file.endswith('.csv'):
            df = load_data(os.path.join(baseline_folder, file))
            baseline_data.append(df)

    user_data = load_data(user_csv_path)

    baseline_poses = split_data_by_pose(pd.concat(baseline_data))
    user_poses = split_data_by_pose(user_data)

    similarities = {}
    for pose in baseline_poses.keys():
        if pose in baseline_poses and pose in user_poses:
            angle_similarities, overall_similarity = calculate_similarity(baseline_poses[pose], user_poses[pose])
            similarities[pose] = (angle_similarities, overall_similarity)

    return similarities

def display_results(similarities, baseline_type):
    print(f"\nResults for {baseline_type} baseline:")
    for pose in golf_pose_order:
        if pose in similarities:
            angle_similarities, overall_similarity = similarities[pose]
            print(f"Pose: {pose}")
            print("Angle similarities (by joint):")
            for joint, similarity in angle_similarities.items():
                print(f"  {joint}: {similarity:.4f}")
            print(f"Overall pose similarity: {overall_similarity:.4f}\n")
        else:
            print(f"Pose: {pose}")
            print("No data available for this pose.\n")

def display_summary(similarities, baseline_type):
    total_similarity = sum(similarities[pose][1] for pose in golf_pose_order if pose in similarities)
    average_similarity = total_similarity / len(golf_pose_order)
    print(f"Summary for {baseline_type} baseline:")
    print(f"Average overall similarity across all poses: {average_similarity:.4f}")
    return average_similarity

def main():
    face_on_similarities = assess_similarity(FACE_ON_BASELINE_FOLDER, USER_CSV_PATH)
    down_the_line_similarities = assess_similarity(DOWN_THE_LINE_BASELINE_FOLDER, USER_CSV_PATH)

    face_on_avg = display_summary(face_on_similarities, "Face-on")
    down_the_line_avg = display_summary(down_the_line_similarities, "Down-the-line")

    print("\nDetailed results:")
    if face_on_avg >= down_the_line_avg:
        display_results(face_on_similarities, "Face-on")
        print("\nThe Face-on baseline provided better results.")
    else:
        display_results(down_the_line_similarities, "Down-the-line")
        print("\nThe Down-the-line baseline provided better results.")

if __name__ == "__main__":
    main()

    print("\nSample data for debugging:")
    face_on_sample = pd.concat([load_data(os.path.join(FACE_ON_BASELINE_FOLDER, f)) for f in os.listdir(FACE_ON_BASELINE_FOLDER) if f.endswith('.csv')]).head()
    down_the_line_sample = pd.concat([load_data(os.path.join(DOWN_THE_LINE_BASELINE_FOLDER, f)) for f in os.listdir(DOWN_THE_LINE_BASELINE_FOLDER) if f.endswith('.csv')]).head()
    user_sample = load_data(USER_CSV_PATH).head()

    print("\nFace-on baseline sample:")
    print(face_on_sample[angle_columns])
    print("\nDown-the-line baseline sample:")
    print(down_the_line_sample[angle_columns])
    print("\nUser sample:")
    print(user_sample[angle_columns])