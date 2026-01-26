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
    
#Charge model
print("Charging the AI model...")
try:
    model = tf.keras.models.load_model('hand_model.h5')
    classes = np.load('classes.npy', allow_pickle=True)
    print("Model charged")
except FileNotFoundError:
    print("CRITICAL ERROR: Files not found")
    print("Execute first src/train_model.py")
    sys.exit()
    
cap = None

for i in range(3):
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        ret, _ = temp_cap.read()
        if ret:
            print(f"[CAMERA] Found in index {i}")
            cap = temp_cap
            break
        else:
            temp_cap.release()
            
if not cap:
    print("CRITICAL ERROR: Camera not found")
    sys.exit()
    
#Variables initialation
detector = HandDetector(max_hands=1)
last_letter_shown = ""
next_letter = ""
FPS_count = 0
THRESHOLD_CONFIRMATION = 8

#'Z' variables for dinamic detection
historial_x = deque(maxlen=15)
Z_THRESHOLD_MOVEMENT = 0.10
z_state = 0
z_last_time = 0
Z_TIMEOUT = 1.5
last_letter = ""

print("--- SIGN LANGUAGE TRANSLATOR ---")
print("Press ESC to exit")

while True:
    success, img = cap.read()
    if not success:
        print("Error to read the camera")
        break
    
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)
    
    actual_letter = ""
    probability = 0.0
    
    if len(lm_list) > 0:
        data_aux = []
        x_base = lm_list[0][1]
        y_base = lm_list[0][2]
        
        h_img, w_img, _ = img.shape
        
        for point in lm_list:
            data_aux.append(point[1] - x_base)
            data_aux.append(point[2] - y_base)
            
        input_data = np.array([data_aux], dtype=np.float32)
        
        #Normalization
        max_val = np.max(np.abs(input_data))
        if max_val != 0:
            input_data = input_data / max_val
        
        prediction = model.predict(input_data, verbose=0)
        max_index = np.argmax(prediction)
        probability = prediction[0][max_index]
        
        detected_class = classes[max_index]
        
        if probability > 0.7:
            
            current_time = time.time()
            
            if z_state > 0 and (current_time - z_last_time > Z_TIMEOUT):
                z_state = 0
                print("Z Timeout Reset")
                
            if detected_class == 'Z_START':
                z_state = 1
                z_last_time = current_time
                last_letter = "Z..."
            elif detected_class == 'Z_MID':
                if z_state == 1:
                    z_state = 2
                    z_last_time = current_time
                elif z_state == 0:
                    z_state = 0
            elif detected_class == 'Z_END':
                if z_state == 2:
                    last_letter = 'Z'
                    z_state = 0
                    print("Z DETECTED!")
                else:
                    z_state = 0
            else:
                if detected_class not in ['Z_START', 'Z_MID', 'Z_END']:
                    z_state = 0
                    actual_letter = detected_class
                    last_letter = actual_letter
                
            cv2.putText(img, f"Letter: {actual_letter} ({int(probability*100)}%)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            if z_state > 0:
                cv2.putText(img, f"Z sequence: step {z_state}/3", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
           
    img_final = cv2.resize(img, (480,320)) 
    cv2.imshow("Raspberry Pi, camera", img_final)
    cv2.moveWindow("Raspberry Pi, camera", 0, 0)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()