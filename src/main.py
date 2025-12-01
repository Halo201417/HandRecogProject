import cv2
import serial
import time
import numpy as np
import os
from detector import HandDetector
import tensorflow as tf

os.environ["QT_QPA_PLATFORM"] = "xcb"

#Starting the AI model
print("Charging the model")
try:
    model = tf.keras.models.load_model('hand_model.h5')
    classes = np.load('classes.npy')
    print("Model charge")
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
            
        input_data = np.array([data_aux], dtype=np.float32)
        
        prediction_probs = model.predict(input_data, verbose=0)
        index_max = np.argmax(prediction_probs)
        probability = prediction_probs[0][index_max]
        
        if probability > 0.8:
            actual_letter = classes[index_max]
            
            cv2.putText(img, f"{actual_letter} ({int(probability*100)}%)", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
        if ser and actual_letter != last_letter:
            ser.write(actual_letter.encode())
            last_letter = actual_letter
            time.sleep(0.1)
            
    cv2.imshow("Neural network", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()
