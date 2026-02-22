import pandas as pd
import numpy as np
import os

#Configuration
SEQUENCE_LENGTH = 30 #How many frames is going to look
INPUT_FILE = 'hand_data.csv'

def convert_static_sequence():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: The {INPUT_FILE} file is not found")
        return
    
    print("Reading static data...")
    df = pd.read_csv(INPUT_FILE)
    
    #Clean the header
    df = df[df['label'] != 'label']
    
    sequences = []
    labels = []
    
    unique_labels = df['label'].unique()
    
    print(f"Transforming {len(unique_labels)} static classes...")
    
    for label in unique_labels:
        label_data = df[df['label'] == label].iloc[:, 1:].values.astype('float32')
        
        #We transform wvery "photo" into a "video"
        for row in label_data:
            seq = np.tile(row, (SEQUENCE_LENGTH, 1))
            sequences.append(seq)
            labels.append(label)
            
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"Done")
    print(f"Format: {X.shape}")
    
    np.save('X_data.npy', X)
    np.save('y_data.npy', y)
    
if __name__ == "__main__":
    convert_static_sequence()