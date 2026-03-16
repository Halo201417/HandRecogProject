import cv2
import time
import numpy as np
import os
from detector import HandDetector
import tensorflow as tf

#Graphic configuration for Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"

#Configuration
SEQUENCE_LENGTH = 30
THRESHOLD = 0.9
COOL_DOWN = 1.5

#Control Commands
CMD_CONFIRM = 'CONFIRM'
CMD_DELETE = 'DELETE'
CMD_CLEAR = 'FINAL'

#Charge LSTM model
print("Charging LSTM model...")
model = tf.keras.models.load_model('hand_model_lstm.h5')
classes = np.load('classes.npy')
detector = HandDetector(max_hands=1)

cap = cv2.VideoCapture(0)
sequence = []
current_letter = ""
word_buffer = []
last_action_time = 0

completed_word_to_display = ""
display_word_start_time = 0
DISPLAY_DURATION = 6.0

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
            
        sequence.append(frame_data)
        sequence = sequence[-SEQUENCE_LENGTH:]
        
        if len(sequence) == SEQUENCE_LENGTH:
            seq_array = np.array(sequence)
            max_val = np.max(np.abs(seq_array))
            
            if max_val == 0:
                max_val = 1.0
                
            seq_normalized = seq_array / max_val
            res = model.predict(np.expand_dims(seq_normalized, axis=0), verbose=0)[0]
            idx = np.argmax(res)
            
            if res[idx] > THRESHOLD:
                prediction = classes[idx]
                current_time = time.time()
                
                if (current_time - last_action_time) > COOL_DOWN:
                    if prediction == CMD_CONFIRM:
                        if current_letter != "" and current_letter not in [CMD_CONFIRM, CMD_DELETE, CMD_CLEAR]:
                            word_buffer.append(current_letter)
                            print(f"Letter '{current_letter}' confirm")
                            last_action_time = current_time
                    elif prediction == CMD_DELETE:
                        if len(word_buffer) > 0:
                            erased = word_buffer.pop()
                            print(f"Letter '{erased}' deleted")
                            last_action_time = current_time
                    elif prediction == CMD_CLEAR:
                        if len(word_buffer) > 0:
                            full_word = "".join(word_buffer)
                            print(f"Complete word: '{full_word}'")

                            completed_word_to_display = full_word
                            display_word_start_time = current_time
                            word_buffer = []
                            last_action_time = current_time
                    else:
                        current_letter = prediction
                        
    else:
        sequence = []
        current_letter = ""
        
    cv2.putText(img, f"Watching: {current_letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    current_word_str = "".join(word_buffer)
    cv2.putText(img, f"Actual Word: {current_word_str}_", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 0), 2)
    
    if (time.time() - last_action_time) < COOL_DOWN:
        cv2.putText(img, "WAIT...", (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(img, "READY", (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    if completed_word_to_display and (time.time() - display_word_start_time) < DISPLAY_DURATION:
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 3
        thickness = 6
        
        text_size = cv2.getTextSize(completed_word_to_display, font, scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        
        cv2.putText(img, completed_word_to_display, (text_x +4, text_y + 4), font, scale, (0, 0, 0), thickness + 4)
        cv2.putText(img, completed_word_to_display, (text_x, text_y), font, scale, (0, 255, 255), thickness)

    cv2.imshow("ASL Translator", img)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
