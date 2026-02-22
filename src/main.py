import cv2
import time
import sys
import numpy as np
import os
from collections import deque
from detector import HandDetector
import tensorflow as tf

#System configuration
RASPBERRY_MODE = True

#Graphic configuration for Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"

#Configuration
SEQUENCE_LENGTH = 30
THRESHOLD = 0.8

#Charge LSTM model
print("Charging LSTM model...")
model = tf.keras.models.load_model('hand_model_lstm.h5')
classes = np.load('classes.npy')
detector = HandDetector(max_hands=1)

cap = cv2.VideoCapture(0)
sequence = []
prediction = ""

while True:
    success, img = cap.read()
    if not success: break
    
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)
    
    if len(lm_list) > 0:
        base_x, base_y = lm_list[0][1], lm_list[0][2]
        frame_data = []
        for point in lm_list:
            frame_data.extend([point[1]- base_x, point[2]- base_y])
            
        max_val = np.max(np.abs(frame_data))
        if max_val == 0:
            max_val = 1.0
        
        frame_data = [val / max_val for val in frame_data]
        
        sequence.append(frame_data)
        sequence = sequence[-SEQUENCE_LENGTH:]
        
        if len(sequence) == SEQUENCE_LENGTH:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            
            if res[np.argmax(res)] > THRESHOLD:
                prediction = classes[np.argmax(res)]
            else:
                prediction = ""
                
            color = (0, 255, 0)
            if prediction == 'Z': color = (0,0,255)
            
            cv2.putText(img, f"Letter: {prediction} ({int(res[np.argmax(res)]*100)}%)", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
    else:
        sequence = []
        
    cv2.imshow("LSTM Translator", img)
    if cv2.waitKey(1) & 0xFF == 27: break
    
cap.release()
cv2.destroyAllWindows()
