import cv2
import numpy as np
import os 
import sys
from detector import HandDetector

# --- General Configuration ---
# Number of frames to form a valid movement
SEQUENCE_LENGTH = 30

# Static Classes Array
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S'
           , 'T', 'U', 'V', 'W', 'X', 'Y']

# Dictionary for the new dynamic collection
KEY_MAP = {
    'z': 'Z',           # Dynamic for the letter z
    'c': 'CONFIRM',     # Gesture to confirm the letter
    'd': 'DELETE',      # Gesture to erase the last letter
    'f': 'FINAL'        # Gesture to show the final word
}

# --- Dataset Initialization ---
# Try to load existing data so we don't overwrite previous records
try:
    X_data = np.load('X_data.npy').tolist()
    y_data = np.load('y_data.npy').tolist()
    print("Files loaded")
except FileNotFoundError:
    # If the files don't exist we start with empty lists
    X_data = []
    y_data = []
    print("Creating dataset")
   
# Initialize webcam 
cap = cv2.VideoCapture(0)
detector = HandDetector(max_hands=1)

# Buffer for the 30 frames to store the movement
sequence_buffer = []

# --- Screen Instructions ---
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
    
    # If a hand is detected on the screen
    if len(lm_list) > 0:
        # Normalization
        # Get the X and Y coordinates of the first point, we use this as
        # an anchor point to normalize the rest of the hand structure
        base_x, base_y = lm_list[0][1], lm_list[0][2]
        frame_data = []
        
        # Iterate through all 21 hand landmarks
        for point in lm_list:
            # Substract the wrist position from every point to get relative
            # coordinates
            frame_data.extend([point[1] - base_x, point[2] - base_y])
        
        # Add this frame 42 relative coordinates (21 points * X,Y) to the buffer    
        sequence_buffer.append(frame_data)
        
        # If the buffer exceeds 30 frames, remove the oldest frame
        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)
    else:
        # Uf the hand leaves the screen we flush the buffer
        sequence_buffer = []
      
    # --- Draw User Interface ---  
    cv2.putText(img, f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(img, f"Data saved: {len(X_data)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    
    cv2.imshow("Sequence Recorder", img)
    
    # Capture key presses 1 ms per frame
    key = cv2.waitKey(1)
    
    # If 'ESC' key is pressed we break the infinite loop
    if key == 27:
        break
    elif key != -1:
        try:
            # Convert the key to lowercase
            char_key = chr(key).lower()
            
            # Check if the key is in our dictionary
            if char_key in KEY_MAP:
                save_label = KEY_MAP[char_key]
                
                # We only save the sequence if the buffer is full
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