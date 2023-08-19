from collections import Counter
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_most_frequent_words(df, num_words=40):
    text = ' '.join(df)    
    words = [word for word in re.findall(r'\w+', text.lower()) if len(word) >= 4]    
    word_counts = Counter(words)    
    most_common_words = word_counts.most_common(num_words)
    return most_common_words

df = pd.read_csv("data/raw/comments.csv", encoding='ISO-8859-2')
print(get_most_frequent_words(df["Comment"]))

# Group by the 'Label' column and count the number of comments for each sentiment class
sentiment_counts = df.groupby('Label').size()

# Calculate percentages for each sentiment
total_comments = len(df)
sentiment_percentages = (sentiment_counts / total_comments) * 100

print("\nCounts of sentiments:")
print(sentiment_counts)
print("\nPercentages of sentiments:")
print(sentiment_percentages)

# Calculate the length of each comment
df['Comment Length'] = df['Comment'].apply(len)
plt.figure(figsize=(10, 6))
for label in df['Label'].unique():
    subset = df[df['Label'] == label]
    plt.hist(subset['Comment Length'], alpha=0.6, label=label, bins=30)
plt.legend(title='Sentiment')
plt.title('Histogram of Comment Lengths by Sentiment Class')
plt.xlabel('Comment Length')
plt.ylabel('Number of Comments')
plt.show()
