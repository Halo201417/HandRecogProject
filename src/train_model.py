import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

physical_devices = tf.config.list_physical_devices('GPU')

if len(physical_devices) > 0:
    print(f"GPU Detected: {physical_devices}")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
else:
    print("Using CPU")
    
def augment_data(X, y, noise_level=0.05, scale_range=(0.9, 1.1), copies=5):
    """
    It generates sinthetic data addind noise and scale.
    This helps the model to not generalize
    """
    
    X_aug, y_aug = [], []
    
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        for _ in range(copies):
            sample = X[i].copy()
            
            #Scale
            scale = np.random.uniform(scale_range[0], scale_range[1])
            sample = sample + scale
            
            #Noise
            noise = np.random.normal(0, noise_level, sample.shape)
            sample = sample + noise
            
            X_aug.append(sample)
            y_aug.append(y[i])
            
    return np.array(X_aug), np.array(y_aug)

print("Loading data...")
try:
    df = pd.read_csv('hand_data.csv')
except FileNotFoundError:
    print("ERROR: File not found")
    exit()
    
X_original = df.iloc[:, 1:].values.astype('float32')
y_strings = df.iloc[:, 0].values

label_encoder = LabelEncoder()
y_index = label_encoder.fit_transform(y_strings)
np.save('classes.npy', label_encoder.classes_)

print(f"Original data: {len(X_original)} samples")
print("Generating Data Augmentation")

X_aug, y_aug = augment_data(X_original, y_index, copies=5)

print(f"Total data for training: {len(X_aug)} samples")

y_categorical = tf.keras.utils.to_categorical(y_aug)

X_train, X_test, y_train, y_test = train_test_split(X_aug, y_categorical, test_size=0.2, random_state=42, stratify=y_aug)

#Model for Raspberry Pi 5
model = Sequential([
    Dense(512, input_shape=(42,), use_bias=False), BatchNormalization(), Activation('relu'), Dropout(0.4),
    Dense(256, use_bias=False), BatchNormalization(), Activation('relu'), Dropout(0.3),
    Dense(128, use_bias=False), BatchNormalization(), Activation('relu'), Dropout(0.3),
    Dense(64, use_bias=False), BatchNormalization(), Activation('relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

#Optimization with learning rate inicial
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=1, min_lr=0.00001),
    EarlyStopping(monitor='val_loss', patience=40, verbose=1, restore_best_weights=True),
    ModelCheckpoint('hand_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

print("Starting intense training...")

history = model.fit(
    X_train, y_train, 
    epochs=300,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

print("Model save as 'hand_model.h5'")

#Export as TFLITE (For Raspberry)
print("Tranforming into TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('hand_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model save as 'hand_model_tflite'")


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy final tests: {accuracy*100:.2f}%")