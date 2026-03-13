#LSTM (Long Short-Term Memory)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

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

def augment_sequence_data(X, y, copies=10, noise_level=0.002, scale_range=(0.85, 1.15)):
    print(f"Creating {copies} copies")
    
    X_aug, y_aug = [], []
    
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        for _ in range(copies):
            sample = X[i].copy()
            
            scale = np.random.uniform(scale_range[0], scale_range[1])
            sample = sample * scale

            noise = np.random.normal(0, noise_level, sample.shape)
            sample = sample + noise
            
            X_aug.append(sample)
            y_aug.append(y[i])
            
    return np.array(X_aug), np.array(y_aug)

print("Charging sequences...")

if os.path.exists('X_data.npy') and os.path.exists('y_data.npy'):
    X_dynamic = np.load('X_data.npy', allow_pickle=True)
    y_dynamic = np.load('y_data.npy', allow_pickle=True)
    
    if y_dynamic.ndim == 0:
        y_dynamic = np.expand_dims(y_dynamic, axis=0)
        
    if X_dynamic.ndim == 2:
        X_dynamic = np.expand_dims(X_dynamic, axis=0)
    
    print(f"Sequences loaded: {X_dynamic.shape[0]}")
else:
    print("Error data not found")
    exit()
    
if os.path.exists('hand_data.csv'):
    df = pd.read_csv('hand_data.csv')
    df = df[df['label'] != 'label']
    
    sequences = []
    labels = []
    unique_labels = df['label'].unique()
    
    for label in unique_labels:
        label_data = df[df['label'] == label].iloc[: , 1:].values.astype('float32')
        for row in label_data:
            seq = np.tile(row,(30,1))
            noise = np.random.normal(loc=0.0, scale=2.0, size=seq.shape)
            seq = seq + noise
            sequences.append(seq)
            labels.append(label)
            
    X_static = np.array(sequences)
    y_static = np.array(labels)
    
    print(f"Static data loaded: {X_static.shape[0]}")
    
X_static = normalize_data(X_static)
X_dynamic = normalize_data(X_dynamic)

X_static_aug , y_static_aug = augment_sequence_data(X_static, y_static, copies=2)
X_dynamic_aug, y_dynamic_aug = augment_sequence_data(X_dynamic, y_dynamic, copies=25)

X = np.concatenate((X_static_aug, X_dynamic_aug), axis=0)
y = np.concatenate((y_static_aug, y_dynamic_aug), axis=0)

print(f"Total saved data for training: {X.shape[0]}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)
np.save('classes.npy', le.classes_)
print(f"Classes detected: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, shuffle=True)

model = Sequential([
    TimeDistributed(Dense(64, activation='relu'), input_shape=(30,42)),
    
    LSTM(128, return_sequences=True),
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

# Model Graph
acc_plot = history.history['accuracy']
val_acc_plot = history.history['val_accuracy']
loss_plot = history.history['loss']
val_loss_plot = history.history['val_loss']
epochs_range = range(len(acc_plot))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc_plot, label='Train Accuracy', color='blue', linewidth=2)
plt.plot(epochs_range, val_acc_plot, label='Validation Accuracy', color='orange', linewidth=2)
plt.title('Evolucion de la Precision')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_plot, label='Train Loss', color='blue', linewidth=2)
plt.plot(epochs_range, val_loss_plot, label='Validation Loss', color='orange', linewidth=2)
plt.title('Evolucion de la Perdida')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_graphs.png', dpi=300)