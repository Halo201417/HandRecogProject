import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y)

#Neuronal network
model = Sequential([
    Dense(256, input_shape=(42,), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True),
    ModelCheckpoint('hand_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

print("Training...")
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

model.save('hand_model.h5')
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy in test: {accuracy*100:.2f}%")
print("Model save")