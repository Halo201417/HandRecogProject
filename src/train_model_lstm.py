#LSTM (Long Short-Term Memory)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

physical_devices = tf.config.list_physical_devices('GPU')

if len(physical_devices) > 0:
    print(f"GPU Detected: {physical_devices}")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
else:
    print("Using CPU")
    
#Configuration
EPOCHS = 500
BATCH_SIZE = 32
PATIENCE = 20
MODEL_NAME = 'hand_model_lstm.h5'

def normalize_data(X):
    X_reshaped = X.reshape((X.shape[0], X.shape[1], 21, 2))
    wrist_coords = X_reshaped[:, :, 0:1, :]
    X_centered = X_reshaped- wrist_coords
    
    max_vals = np.max(np.abs(X_centered), axis=(2,3), keepdims=True)
    max_vals[max_vals == 0] = 1.0
    X_scaled = X_centered / max_vals
    
    return X_scaled.reshape((X.shape[0], X.shape[1], 42))

print("Charging sequences...")
if not os.path.exists('X_data.npy') or not os.path.exists('y_data.npy'):
    print("ERROR: Files not found")
    exit()
    
X = np.load('X_data.npy')
y = np.load('y_data.npy')

print(f"Saved data: {X.shape[0]}")

if X.shape[0] == 0:
    print("Error, no data")
    exit()

X = normalize_data(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)
np.save('classes.npy', le.classes_)
print(f"Classes detected: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, shuffle=True)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30,42)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(MODEL_NAME, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
]

print(f"Starting training....")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

print("Training finalizate")
print(f"Model saved {MODEL_NAME}")

loss, acc = model.evaluate(X_test, y_test)
print(f"Precission final test: {acc*100:.2f}%")