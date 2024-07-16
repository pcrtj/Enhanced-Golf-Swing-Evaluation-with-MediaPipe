import pandas as pd

csv_path = './golfdb/data/GolfDB.csv'

df = pd.read_csv(csv_path)

print(df.columns)
