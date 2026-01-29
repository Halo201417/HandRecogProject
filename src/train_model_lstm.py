#LSTM (Long Short-Term Memory)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("Charging sequences...")
X = np.load('X_data.npy')
y = np.load('y_data.npy')

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)
np.save('classes.npy', le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

#Model LSTM
model = Sequential([
    LSTM(64, return_sequences=False, activation='relu', input_shape=(30,42)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model....")
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))

model.save('hand_model_lstm.h5')
print("Model saved")