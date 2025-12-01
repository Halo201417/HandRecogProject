import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Loading data...")
try:
    df = pd.read_csv('hand_data.csv')
except FileNotFoundError:
    print("ERROR: File not found")
    exit()
    
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

print("Training the model (this can last several minutes...)")
model = RandomForestClassifier(n_estimators=100, criterion='gini')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained. Accuracy: {accuracy * 100:.2f}%")

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
    
print("Model saved successfully")