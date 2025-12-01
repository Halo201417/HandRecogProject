import cv2
import csv
import os
from detector import HandDetector

FILE_NAME = 'hand_data.csv'

cap = cv2.VideoCapture(0)
detector = HandDetector(max_hands=1)

if not os.path.exists(FILE_NAME):
    with open(FILE_NAME, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        
        for i in range(21):
            header.extend([f'x{i}', f'y{i}'])
            
        writer.writerow(header)
        
print("Press a key (a-z) to save the gesture")
print("Press ESC to exit")

while True:
    success, img = cap.read()
    
    if not success:
        continue
    
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)
    
    cv2.putText(img, "Press a letter to save", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Data collection", img)
    key = cv2.waitKey(1)
    
    if key == 27:
        break
    elif key != -1 and len(lm_list) > 0:
        char_pressed = chr(key).upper()
        
        if 'A' <= char_pressed <= 'Z':
            base_x, base_y = lm_list[0][1], lm_list[0][2]
            row = [char_pressed]
            
            for point in lm_list:
                row.extend([point[1] - base_x, point[2] - base_y])
                
            with open(FILE_NAME, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            print(f"Sampled of the letter saved: {char_pressed}")
            
cap.release()
cv2.destroyAllWindows()