import cv2
import csv
import os
import sys
from detector import HandDetector

# --- General Configuration ---
# Target file where the "photos" are goingo to be stored
FILE_NAME = 'hand_data.csv'

# --- Hardware Initialization ---
# Camera detection loop, it checks the first 3 indices (0, 1, 2)
# to find an available camera. This prevents crashes if /dev/video0
# is busy or another camera is connected 
cap = None
for i in range(3):
    try:
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            # Test if we can grab a frame succesfully
            ret, _ = temp_cap.read()
            if ret:
                cap = temp_cap
                print(f"[Info] Camera detected in index {i}")
                break
            else:
                temp_cap.release()
    except:
        pass
    
if not cap:
    print("ERROR: Camera not detected")
    sys.exit()
    
# Initialize the hand tracking module
detector = HandDetector(max_hands=1)

# --- Dataset Preparation ---
# Create a CSV file and write the header row if it doesn't already exist
if not os.path.exists(FILE_NAME):
    with open(FILE_NAME, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header format: label, x0, t0, x1, t1, .... until 20
        # t represetns the Y coordinate axis
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f't{i}'])
        
        writer.writerow(header)

# --- Screen Instructions ---        
print("----DATA INSTRUCTIONS----")
print("1. Do NOT save the letter Z here")
print("2. Save the D letter (Index finger up)")
print("3. The letter Z is a special case, we need to do it in 3 parts")
print("Press ESC to exit")

while True:
    success, img = cap.read()
    
    if not success:
        continue
    
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)
    
    # Draw user interface
    cv2.putText(img, "Press key to save (A-Y)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, "DO NOT SAVE THE LETTER Z THE CONVENTIONAL WAY", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Data collection", img)
    
    # Capture key presses 1 ms per frame
    key = cv2.waitKey(1)
    
    # If 'ESC' key is pressed we break the infinite loop
    if key == 27:
        break
    elif key != -1 and len(lm_list) > 0:
        
        label_to_save = None
        
        # Z logic not suitable
        if key == ord('1'):
            label_to_save = 'Z_START'
        elif key == ord('2'):
            label_to_save = 'Z_MID'
        elif key == ord('3'):
            label_to_save = 'Z_END'
        else:
            try:
                char_pressed = chr(key).upper()
                
                # Standard alphabet mapping
                if 'A' <= char_pressed <= 'Y':
                    label_to_save = char_pressed
                elif char_pressed == 'Z':
                    print("WARNING: Do not Press Z. Use keys 1, 2, 3")
            except ValueError:
                pass
        
        # --- Save data to CSV ---
        if label_to_save:
            # Normalization
            # Set landmark 0 as the central anchor point
            base_x, base_y = lm_list[0][1], lm_list[0][2]
            row = [label_to_save]
            
            # Substract the wrist position from every point to get relative
            # coordinates
            for point in lm_list:
                row.extend([point[1] - base_x, point[2] - base_y])
            
            # Open the CSV in append mode to add the new row without deleting old data    
            with open(FILE_NAME, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            print(f"Data saved: {label_to_save}")
    
            
cap.release()
cv2.destroyAllWindows()