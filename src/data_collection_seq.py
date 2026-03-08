import cv2
import numpy as np
import os 
import sys
from detector import HandDetector

#Configuration
SEQUENCE_LENGTH = 30
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S'
           , 'T', 'U', 'V', 'W', 'X', 'Y']
KEY_MAP = {
    'z': 'Z',
    'c': 'CONFIRM',
    'd': 'DELETE',
    'f': 'FINAL'
}

try:
    X_data = np.load('X_data.npy').tolist()
    y_data = np.load('y_data.npy').tolist()
    print("Files loaded")
except FileNotFoundError:
    X_data = []
    y_data = []
    print("Creating dataset")
    
cap = cv2.VideoCapture(0)
detector = HandDetector(max_hands=1)

sequence_buffer = []    #Last 30 frames here

print(f"SEQUENCE RECORDER")
print("1. Make the movement")
for key_char, label in KEY_MAP.items():
    print(f"2 -> Press '{key_char}' to save: {label}")
print("3. ESC to exit and save")

while True:
    success, img = cap.read()
    if not success: break
    
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)
    
    if len(lm_list) > 0:
        base_x, base_y = lm_list[0][1], lm_list[0][2]
        frame_data = []
        
        for point in lm_list:
            frame_data.extend([point[1] - base_x, point[2] - base_y])
            
        sequence_buffer.append(frame_data)
        
        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)
    else:
        sequence_buffer = []
        
    cv2.putText(img, f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(img, f"Data saved: {len(X_data)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    
    cv2.imshow("Sequence Recorder", img)
    
    key = cv2.waitKey(1)
    
    if key == 27:
        break
    elif key != -1:
        try:
            char_key = chr(key).lower()
            if char_key in KEY_MAP:
                save_label = KEY_MAP[char_key]
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    X_data.append(np.array(sequence_buffer))
                    y_data.append(save_label)
                    print(f"Sequence of {save_label} saved")
                else:
                    print(f"Incomplete buffer for {save_label}, make the movement slower")
        except ValueError:
            pass
            
print("Saving files...")
np.save('X_data.npy', np.array(X_data))
np.save('y_data.npy', np.array(y_data))
print("Saved")
cap.release()
cv2.destroyAllWindows()