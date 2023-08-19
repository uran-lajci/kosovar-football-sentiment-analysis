import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv('data/preprocessed/preprocessed_train_data.csv')
test_data = pd.read_csv('data/preprocessed/preprocessed_test_data.csv')

exclude_cols = ['Game ID', 'Comment', 'Label', 'Home Team', 'Away Team', 'Kosovas Result', 'Processed_Comments']

X_train = train_data.drop(columns=exclude_cols)
y_train = train_data['Label']
X_val = test_data.drop(columns=exclude_cols)
y_val = test_data['Label']

models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=10000), 
    'Naive Bayes': GaussianNB()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    class_report = classification_report(y_val, y_pred)
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(class_report)
    print("-" * 60)
