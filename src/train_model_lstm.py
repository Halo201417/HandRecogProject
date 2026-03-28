# ---LSTM (Long Short-Term Memory) Model ---
# LSTM is a specialized type of Recurrent Neural Network (RNN) designed specially
# to understand sequences, patterns and time-series data.

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

# Check if an NVIDIA GPU is available to accelerate training
physical_devices = tf.config.list_physical_devices('GPU')

if len(physical_devices) > 0:
    print(f"GPU Detected: {physical_devices}")
    try:
        # Prevent TensorFlow from allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
else:
    print("Using CPU")
    
# --- Training Parameters ---
EPOCHS = 500        # Maximum number of traning iterations
BATCH_SIZE = 32     # Number of samples processed before updating weights
PATIENCE = 20       # Stop training if no improvement after 20 epochs
MODEL_NAME = 'hand_model_lstm.h5'

def normalize_data(X):
    """
    Normalizes the hand tracking data geometrically and by magnitude
    """
    
    # Reshape from flat 42 numbers to 21 pairs of (X, Y) coordinates
    X_reshaped = X.reshape((X.shape[0], X.shape[1], 21, 2))
    
    # Subtract the wrist coordinates (point 0) from all other points
    wrist_coords = X_reshaped[:, :, 0:1, :]
    X_centered = X_reshaped- wrist_coords
    
    # Find the maximum coordinate value in each frame
    max_vals = np.max(np.abs(X_centered), axis=(2,3), keepdims=True)
    max_vals[max_vals == 0] = 1.0   # Prevent division by zero
    
    # Scalle all coordinates to a value between -1 and 1
    X_scaled = X_centered / max_vals
    
    # Flatten back to the original 42-number format required by the Dense Layer
    return X_scaled.reshape((X.shape[0], X.shape[1], 42))

def augment_sequence_data(X, y, copies=10, noise_level=0.002, scale_range=(0.85, 1.15)):
    """
    Artificially increases the dataset size to prevent overfitting.
    It creates slightly modified clones of the original gestures
    """
    
    print(f"Creating {copies} copies")
    
    X_aug, y_aug = [], []
    
    for i in range(len(X)):
        # Always keep the original unmodified
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Create copies
        for _ in range(copies):
            sample = X[i].copy()
            
            # Random scaling
            scale = np.random.uniform(scale_range[0], scale_range[1])
            sample = sample * scale

            # Random noise
            noise = np.random.normal(0, noise_level, sample.shape)
            sample = sample + noise
            
            X_aug.append(sample)
            y_aug.append(y[i])
            
    return np.array(X_aug), np.array(y_aug)

print("Loading sequences...")

# Load Dynamic Gestures
if os.path.exists('X_data.npy') and os.path.exists('y_data.npy'):
    X_dynamic = np.load('X_data.npy', allow_pickle=True)
    y_dynamic = np.load('y_data.npy', allow_pickle=True)
    
    # Fix dimensions if the dataset only contains 1 sample
    if y_dynamic.ndim == 0:
        y_dynamic = np.expand_dims(y_dynamic, axis=0)
        
    if X_dynamic.ndim == 2:
        X_dynamic = np.expand_dims(X_dynamic, axis=0)
    
    print(f"Sequences loaded: {X_dynamic.shape[0]}")
else:
    print("Error data not found")
    exit()
 
# Load Static Gestures   
if os.path.exists('hand_data.csv'):
    df = pd.read_csv('hand_data.csv')
    df = df[df['label'] != 'label'] # Clean up accidental headers
    
    sequences = []
    labels = []
    unique_labels = df['label'].unique()
    
    for label in unique_labels:
        label_data = df[df['label'] == label].iloc[: , 1:].values.astype('float32')
        for row in label_data:
            
            # Transform static images into 30-frame video sequence
            seq = np.tile(row,(30,1))
            
            # Add heavier noise to static data to make it look dynamic
            noise = np.random.normal(loc=0.0, scale=2.0, size=seq.shape)
            seq = seq + noise
            
            sequences.append(seq)
            labels.append(label)
            
    X_static = np.array(sequences)
    y_static = np.array(labels)
    
    print(f"Static data loaded: {X_static.shape[0]}")

# Normalize both datasets   
X_static = normalize_data(X_static)
X_dynamic = normalize_data(X_dynamic)

# We create more copies of dynamic data because static data has more initial samples
# Change on need
X_static_aug , y_static_aug = augment_sequence_data(X_static, y_static, copies=2)
X_dynamic_aug, y_dynamic_aug = augment_sequence_data(X_dynamic, y_dynamic, copies=25)

# Merge both datasets into a single one
X = np.concatenate((X_static_aug, X_dynamic_aug), axis=0)
y = np.concatenate((y_static_aug, y_dynamic_aug), axis=0)

print(f"Total saved data for training: {X.shape[0]}")

# Encode text labels into numerical categorical data
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# Save the dictionary of classes so the main script knows what number means what letter
np.save('classes.npy', le.classes_)
print(f"Classes detected: {le.classes_}")

# 80% for training, 20% for testing and validation
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, shuffle=True)

# --- Neural Network Architecture ---
model = Sequential([
    # TimeDistributed applies the Dense layer to every single frame individually
    # before passing it to the LSTM, it helps extract fundamental features from the
    # 42 coordinates in each frame
    TimeDistributed(Dense(64, activation='relu'), input_shape=(30,42)),
    
    # LSTM Layers are used to understand temporal sequences and moving data
    LSTM(128, return_sequences=True),
    Dropout(0.2),   # Drops 20% of neurons randomly to prevent memorization
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    
    # Fully Connected Layers to classify the final result
    Dense(64, activation='relu'),
    BatchNormalization(),   # Stabilizes and accelerates the learning process
    Dropout(0.2),
    Dense(32, activation='relu'),
    
    # Output Layer: Softmax outputs a probability percentage for each class
    Dense(len(le.classes_), activation='softmax')
])

# Use Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Callbacks and Training ---
callbacks = [
    # Only save the model file if the validation loss improves
    ModelCheckpoint(MODEL_NAME, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
    # Srop trainig early if the AI stops improving after 20 epochs
    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    # If learning decay, reduce the learning rate by half to make smaller and finer adjustments
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

# Evaluate the final accuracy on the 20% testing data
loss, acc = model.evaluate(X_test, y_test)
print(f"Precission final test: {acc*100:.2f}%")

# --- Graphs ---
acc_plot = history.history['accuracy']
val_acc_plot = history.history['val_accuracy']
loss_plot = history.history['loss']
val_loss_plot = history.history['val_loss']
epochs_range = range(len(acc_plot))

# Accuracy
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc_plot, label='Train Accuracy', color='blue', linewidth=2)
plt.plot(epochs_range, val_acc_plot, label='Validation Accuracy', color='orange', linewidth=2)
plt.title('Evolucion de la Precision')
plt.legend(loc='lower right')
plt.grid(True)

# Loss (Error rate)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_plot, label='Train Loss', color='blue', linewidth=2)
plt.plot(epochs_range, val_loss_plot, label='Validation Loss', color='orange', linewidth=2)
plt.title('Evolucion de la Perdida')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_graphs.png', dpi=300)