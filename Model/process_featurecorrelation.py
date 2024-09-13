import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# กำหนด path ของโฟลเดอร์ที่มีไฟล์ CSV
CSV_FOLDER = "./output/videos_raw/csv/combined/realtime"

# อ่านทุกไฟล์ CSV ในโฟลเดอร์
all_data = []
for filename in os.listdir(CSV_FOLDER):
    if filename.endswith(".csv"):
        file_path = os.path.join(CSV_FOLDER, filename)
        df = pd.read_csv(file_path)
        
        # แยกคอลัมน์ x, y
        for joint in ['Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Wrist', 'Right Wrist', 'Left Ankle', 'Right Ankle', 'Nose']:
            df[[f'{joint} x', f'{joint} y']] = df[f'x, y {joint}'].str.split(', ', expand=True).astype(float)
        
        all_data.append(df)

# รวมข้อมูลจากทุกไฟล์
combined_df = pd.concat(all_data, ignore_index=True)

# เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข
numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
numeric_df = combined_df[numeric_columns]

# คำนวณ correlation matrix
corr_matrix = numeric_df.corr()

# สร้าง heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', vmin=-1, vmax=1, center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# แสดงค่า correlation ที่สูงที่สุด (ไม่รวมค่า 1.0 ที่เป็น correlation กับตัวเอง)
high_corr = corr_matrix.unstack()
high_corr = high_corr[high_corr < 1.0]
high_corr = high_corr.sort_values(ascending=False)
print("Top 10 highest correlations:")
print(high_corr.head(10))

# แสดงค่า correlation ที่ต่ำที่สุด
print("\nTop 10 lowest correlations:")
print(high_corr.tail(10))

# หาคุณลักษณะที่มีความสัมพันธ์สูงกับตัวแปรเป้าหมาย (สมมติว่า 'Pose' เป็นตัวแปรเป้าหมาย)
# หมายเหตุ: เนื่องจาก 'Pose' เป็นตัวแปรเชิงคุณภาพ เราจะไม่สามารถคำนวณ correlation โดยตรงได้
# แทนที่จะใช้ 'Pose' เราจะใช้คอลัมน์ตัวเลขอื่นเป็นตัวอย่าง เช่น 'Left Shoulder Angle'
target_column = 'Left Shoulder Angle'
cor_target = abs(corr_matrix[target_column])

# เลือกคุณลักษณะที่มีค่า correlation สูงกว่า 0.5
relevant_features = cor_target[cor_target > 0.5]
print(f"\nFeatures highly correlated with {target_column} (correlation > 0.5):")
print(relevant_features)

# สร้าง heatmap เฉพาะคุณลักษณะที่สำคัญ
important_features = list(relevant_features.index)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix.loc[important_features, important_features], 
            annot=True, cmap='RdYlBu_r', vmin=-1, vmax=1, center=0, fmt='.2f')
plt.title(f'Correlation Heatmap of Features Highly Correlated with {target_column}')
plt.tight_layout()
plt.savefig('important_features_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()