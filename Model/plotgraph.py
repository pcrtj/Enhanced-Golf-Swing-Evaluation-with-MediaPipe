import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Read data from the CSV file
file_path = './output/videos_keyframedetection/raw_data/hpe_raw_data/csv/keyframe_1_angles.csv'
data = pd.read_csv(file_path)

# Points to plot
points = [
    'x, y Left Shoulder', 'x, y Right Shoulder', 'x, y Left Elbow', 'x, y Right Elbow',
    'x, y Left Hip', 'x, y Right Hip', 'x, y Left Knee', 'x, y Right Knee',
    'x, y Left Wrist', 'x, y Right Wrist', 'x, y Left Ankle', 'x, y Right Ankle', 'x, y Nose'
]

# Create a figure and 3D axis for plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot positions x, y, and time (z) of various points
for point in points:
    col = f'{point}'
    if col in data.columns:
        # Separate x and y values
        x_values = []
        y_values = []
        z_values = data['Time']  # Time values as z-axis
        for value in data[col]:
            x, y = map(float, value.split(','))
            x_values.append(x)
            y_values.append(y)
        ax.scatter(x_values, y_values, z_values, label=point)
    else:
        print(f"Missing column: {col}")

# Add labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Time')
ax.set_title('Position of Various Points Over Time')
ax.legend()
ax.grid(True)

# Show plot
plt.show()
