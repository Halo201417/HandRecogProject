import cv2
import serial
import time
import sys
import pickle
import numpy as np
from detector import HandDetector
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

#Starting the AI model
print("Charging the model")
try:
    with open('model.p', 'rb') as f:
        model_data = pickle.load(f)
        
    model = model_data['model']
    print("Model loaded")
except FileNotFoundError:
    print("ERROR: File not Found")
    sys.exit()

# Configuration for the serial port (in Linux /dev/ttyACM0)
try:
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)   #Wait for a reseat
    print("Connection successful")
except:
    print("WARNING: Arduino is not connected")
    ser = None

# Configuration of camera and Detector
cap = None

for index in range(3):
    print(f"Testing camara index {index}...")
    temp_cap = cv2.VideoCapture(index)
    
    if temp_cap.isOpened():
        ret, frame = temp_cap.read()
        
        if ret:
            print(f"[SUCCESS] Camera found in the index {index}")
            cap = temp_cap
            break
        else:
            temp_cap.release()
    else:
        temp_cap.release()
        
if cap is None:
    print("Critical error: Not camera found")
    sys.exit()
    
detector = HandDetector(max_hands=1)
last_prediction = ""

print("System ready")
print("Press 'Esc' to exit")

while True:
    success, img = cap.read()
    
    if not success:
        print("Error reading frame")
        continue
    
    #Find hands
    img = detector.find_hands(img)
    
    #Find positions
    lm_list = detector.find_position(img)
    
    if len(lm_list) != 0:
        data_aux = []
        x_base = lm_list[0][1]
        y_base = lm_list[0][2]
        
        for point in lm_list:
            data_aux.append(point[1] - x_base)
            data_aux.append(point[2] - y_base)
            
        prediction = model.predict([np.asarray(data_aux)])
        letter_detected = prediction[0]
        
        cv2.rectangle(img, (0,0), (160,160), (0,0,0), -1)
        cv2.putText(img, f"Letter: {letter_detected}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        if ser and letter_detected != last_prediction:
            ser.write(letter_detected.encode())
            last_prediction = letter_detected
            print(f"Send: {letter_detected}")
            time.sleep(0.1)
            
    cv2.imshow("Translate LSE", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()
