import cv2
import serial
import time
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
cap = cv2.VideoCapture(2)
detector = HandDetector(max_hands=1)

while True:
    success, img = cap.read()
    if not success:
        continue
    
    #Find hand
    img = detector.find_hands(img)
    
    #Find positions
    lm_list = detector.find_position(img)
    
    if len(lm_list) != 0:
        fingers = detector.count_fingers()
        
        total_fingers = fingers.count(1)
        print(f"Fingers detected: {total_fingers}")
        
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