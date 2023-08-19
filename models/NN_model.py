from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from keras.utils import to_categorical

import pandas as pd

train_data = pd.read_csv('data/preprocessed/preprocessed_train_data.csv')
test_data = pd.read_csv('data/preprocessed/preprocessed_test_data.csv')

exclude_cols = ['Game ID', 'Comment', 'Label', 'Home Team', 'Away Team', 'Kosovas Result', 'Processed_Comments']
X_train = train_data.drop(columns=exclude_cols)
y_train_labels = train_data['Label']

X_test = test_data.drop(columns=exclude_cols)
y_test_labels = test_data['Label']

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_labels)
y_test_encoded = label_encoder.transform(y_test_labels)

y_train = to_categorical(y_train_encoded)
y_test = to_categorical(y_test_encoded)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax')) 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Validation Accuracy: {accuracy*100:.2f}%")