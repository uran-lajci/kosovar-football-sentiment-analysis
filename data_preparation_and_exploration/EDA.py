import pandas as pd

df = pd.read_csv("comments.csv", encoding='ISO-8859-2')

print(df.head())

print(df.info())

print(df["Label"].value_counts())

grouped = df.groupby(['Label', 'Kosovas Result']).size().reset_index(name='count')
print(grouped)