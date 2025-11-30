import cv2
import serial
import time
import sys
from detector import HandDetector
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

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
        fingers = detector.count_fingers()
        total_fingers = fingers.count(1)
        
        print(f"Fingers: {total_fingers}")
        
        if ser:
            if total_fingers == 5:
                ser.write(b'O')
            elif total_fingers == 0:
                ser.write(b'C')       
    else:
        if ser:
            ser.write(b'N')
            
    cv2.imshow("Hand Recognition", img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()
