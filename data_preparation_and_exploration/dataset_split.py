from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_csv("comments.csv", encoding='ISO-8859-2')

train_data, test_data = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)

print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)