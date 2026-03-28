import cv2
import time
import numpy as np
import os
from detector import HandDetector
import tensorflow as tf

# --- Environment Configuration ---
# Fix for Qt GUI errors in some Linux environments
os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- Translator Configuration ---
SEQUENCE_LENGTH = 30    # Number of frames the LSTM model expects per prediction
THRESHOLD = 0.9         # Minimum confidence level to accept a prediction
COOL_DOWN = 1.5         # Wait time (seconds) between confirming letters

# Dynamic system commands
CMD_CONFIRM = 'CONFIRM'
CMD_DELETE = 'DELETE'
CMD_CLEAR = 'FINAL'

print("Charging LSTM model...")
# Load the trained NN using TensorFlow
model = tf.keras.models.load_model('hand_model_lstm.h5')

# Load the ordered list of classes
classes = np.load('classes.npy')

# Initialize the hand tracking module
detector = HandDetector(max_hands=1)

# Initialize the webcam
cap = cv2.VideoCapture(0)
sequence = []   # Sliding window buffer

# --- Natural Language Processing Variables ---
current_letter = ""     # The letter currently held
word_buffer = []        # Array accumulating the confirmed letters to form a word
last_action_time = 0    # Timestamp of the last action to manage the cooldown timer

# --- Visual Effect Variables ---
completed_word_to_display = ""
display_word_start_time = 0
DISPLAY_DURATION = 6.0  # How long the giant completed word stays on screen

while True:
    success, img = cap.read()
    if not success: break
    
    # Detect hands and extract bone landmarks
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)
    
    if len(lm_list) > 0:
        # Normalization
        # Use the wrist as the central anchor
        base_x, base_y = lm_list[0][1], lm_list[0][2]
        frame_data = []
        
        # Convert absolute screen coordinates to relative distances from the wrist
        for point in lm_list:
            frame_data.extend([point[1] - base_x, point[2] - base_y])
         
        # Append the 42 coordinates of the current frame   
        sequence.append(frame_data)
        
        # Keep the buffer strictly at the required length
        sequence = sequence[-SEQUENCE_LENGTH:]
        
        # Once we have enough frames to form a complete video sequence
        if len(sequence) == SEQUENCE_LENGTH:
            seq_array = np.array(sequence)
            
            # Normalization
            # Find the maximum absolute value in the array to scale everything between
            # -1 and 1
            max_val = np.max(np.abs(seq_array))
            if max_val == 0: max_val = 1.0                
            seq_normalized = seq_array / max_val
            
            # Ask the NN to predict the gesture
            # np.expand_dims adds a batch dimension
            res = model.predict(np.expand_dims(seq_normalized, axis=0), verbose=0)[0]
            idx = np.argmax(res)    # Get the index of the highest probability
            
            
            if res[idx] > THRESHOLD:
                prediction = classes[idx]
                current_time = time.time()
                
                # Only allow registering a new command if the cooldown period has passed
                if (current_time - last_action_time) > COOL_DOWN:
                    
                    # Command CONFIRM: add the current letter
                    if prediction == CMD_CONFIRM:
                        if current_letter != "" and current_letter not in [CMD_CONFIRM, CMD_DELETE, CMD_CLEAR]:
                            word_buffer.append(current_letter)
                            print(f"Letter '{current_letter}' confirm")
                            last_action_time = current_time
                            
                    # Command DELETE: remove the last added letter
                    elif prediction == CMD_DELETE:
                        if len(word_buffer) > 0:
                            erased = word_buffer.pop()
                            print(f"Letter '{erased}' deleted")
                            last_action_time = current_time
                            
                    # Command FINAL: finish the word, display it and flush the buffer
                    elif prediction == CMD_CLEAR:
                        if len(word_buffer) > 0:
                            full_word = "".join(word_buffer)
                            print(f"Complete word: '{full_word}'")

                            # Trigger the visual text timer
                            completed_word_to_display = full_word
                            display_word_start_time = current_time
                            
                            # Empty the list for the next word
                            word_buffer = []
                            last_action_time = current_time
                    # Standard letter prediction
                    else:
                        current_letter = prediction
                        
    else:
        # If the hand leaves the screen, flush the buffer and reset the current letter
        sequence = []
        current_letter = ""
    
    
    # --- Draw Standard User Interface ---    
    cv2.putText(img, f"Watching: {current_letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    current_word_str = "".join(word_buffer)
    cv2.putText(img, f"Actual Word: {current_word_str}_", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 0), 2)
    
    # Display cooldown status
    if (time.time() - last_action_time) < COOL_DOWN:
        cv2.putText(img, "WAIT...", (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(img, "READY", (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
     
    # --- Draw Gigant Completed Word ---   
    if completed_word_to_display and (time.time() - display_word_start_time) < DISPLAY_DURATION:
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 3
        thickness = 6
        
        # Calculate the text size to center it on the screen
        text_size = cv2.getTextSize(completed_word_to_display, font, scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        
        # Draw a black shadow underneath
        cv2.putText(img, completed_word_to_display, (text_x +4, text_y + 4), font, scale, (0, 0, 0), thickness + 4)
        
        # Draw the main text on top
        cv2.putText(img, completed_word_to_display, (text_x, text_y), font, scale, (0, 255, 255), thickness)

    # Show the final compiled frame
    cv2.imshow("ASL Translator", img)
    
    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
