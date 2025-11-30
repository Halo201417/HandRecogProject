import cv2
import mediapipe as mp
import serial
import time

# Configuration for the serial port (in Linux /dev/ttyACM0)
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)   #Wait for a reseat

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            continue
        
        #Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        if results.multi_hand_landmarks:
            print("Hand detected -> send to arduino")
            ser.write(b'H')
        else:
            ser.write(b'N')
            
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Recognition', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
cap.release()
ser.close()