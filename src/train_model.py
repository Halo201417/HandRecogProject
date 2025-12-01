import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("Loading data...")
try:
    df = pd.read_csv('hand_data.csv')
except FileNotFoundError:
    print("ERROR: File not found")
    exit()
    
X = df.iloc[:, 1:].values.astype('float32')
y_strings = df.iloc[:, 0].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_strings)

np.save('classes.npy', label_encoder.classes_)

y_categorical = tf.keras.utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

#Neuronal network
model = Sequential([
    Dense(128, input_shape=(42,), activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

model.save('hand_model.h5')
print("Model save")