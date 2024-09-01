import pandas as pd

# โหลดข้อมูล
golfdb = pd.read_csv('./golfdb/data/GolfDB.csv')

# พิมพ์ชื่อคอลัมน์ทั้งหมด
print(golfdb.columns)
