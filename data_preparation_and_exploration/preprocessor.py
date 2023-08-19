import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('romanian') 

albanian_stopwords= ["dhe", "në", "e", "të", "i", "një", "me", "për", "nga", "që", "ka", "është"]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in albanian_stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    text = ' '.join(tokens)
    return text

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

train_data['Processed_Comments'] = train_data['Comment'].apply(preprocess_text)
test_data['Processed_Comments'] = test_data['Comment'].apply(preprocess_text)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['Processed_Comments'])
X_test = vectorizer.transform(test_data['Processed_Comments'])

train_bow_df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
test_bow_df = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())

train_data = pd.concat([train_data, train_bow_df], axis=1)
test_data = pd.concat([test_data, test_bow_df], axis=1)

train_data.to_csv('preprocessed_train_data.csv', index=False)
test_data.to_csv('preprocessed_test_data.csv', index=False)