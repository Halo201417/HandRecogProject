import pandas as pd
import numpy as np
import os

# --- General Configuration ---
# SEQUENCE_LENGTH defines how many frames the LSTM expects
SEQUENCE_LENGTH = 30
INPUT_FILE = 'hand_data.csv'

def convert_static_sequence():
    
    """
    Reads static hand landmarks saved in the CSV and transforms them
    into repeated sequences so they are compatible with the dynamic
    LSTM model
    """
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: The {INPUT_FILE} file is not found")
        return
    
    print("Reading static data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Clean the header
    df = df[df['label'] != 'label']
    
    sequences = []  # Main array for the "videos"
    labels = [] # Array to store the letters
    
    unique_labels = df['label'].unique()
    
    print(f"Transforming {len(unique_labels)} static classes...")
    
    for label in unique_labels:
        # 1. Filter rows for a specific letter
        # 2. Then it skips the first column keeping only the numbers
        # 3. Convert number to float32 format
        label_data = df[df['label'] == label].iloc[:, 1:].values.astype('float32')
        
        for row in label_data:
            # We transform every single "photo" into a "video"
            # np.tile copies and pastes the exact same row 30 times 
            seq = np.tile(row, (SEQUENCE_LENGTH, 1))
            sequences.append(seq)
            labels.append(label)
            
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"Done")
    print(f"Format: {X.shape}")
    
    # Transforms the matrices into binary
    np.save('X_data.npy', X)
    np.save('y_data.npy', y)
    
if __name__ == "__main__":
    convert_static_sequence()