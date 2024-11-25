import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_sequences(sequences):
    """Convert DNA sequences to integer encodings.
    
    Args:
        sequences: Array of DNA sequences
    
    Returns:
        numpy.ndarray: Array of encoded sequences
    """
    nuc_map = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}
    processed_seqs = []
    for seq in sequences:
        processed_seq = [nuc_map.get(nuc, 0) for nuc in seq.upper()]
        processed_seqs.append(processed_seq)
    return np.array(processed_seqs)

def load_and_preprocess_data(file_path):
    """Load and preprocess sgRNA data from file.
    
    Args:
        file_path: Path to CSV file containing sequences and scores
        
    Returns:
        tuple: Training and validation data splits
    """
    data = pd.read_csv(file_path)
    data = data.dropna() 
    sequences = data['sequence'].values
    scores = data['score'].values
    
    X = preprocess_sequences(sequences)
    y = scores
    
    return train_test_split(X, y, test_size=0.2, random_state=42)