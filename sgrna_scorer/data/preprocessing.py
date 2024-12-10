import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataNormalizer:
    def __init__(self):
        self.sequence_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def fit_transform(self, sequences, targets):
        """Fit and transform both sequences and targets."""
        # Reshape sequences for StandardScaler (2D array required)
        seq_shape = sequences.shape
        reshaped_seqs = sequences.reshape(-1, seq_shape[-1])
        
        # Fit and transform sequences
        normalized_seqs = self.sequence_scaler.fit_transform(reshaped_seqs)
        normalized_seqs = normalized_seqs.reshape(seq_shape)
        
        # Fit and transform targets
        normalized_targets = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        return normalized_seqs, normalized_targets
    
    def transform(self, sequences, targets):
        """Transform new sequences and targets using fitted scalers."""
        # Reshape sequences for StandardScaler
        seq_shape = sequences.shape
        reshaped_seqs = sequences.reshape(-1, seq_shape[-1])
        
        # Transform sequences
        normalized_seqs = self.sequence_scaler.transform(reshaped_seqs)
        normalized_seqs = normalized_seqs.reshape(seq_shape)
        
        # Transform targets
        normalized_targets = self.target_scaler.transform(targets.reshape(-1, 1)).flatten()
        
        return normalized_seqs, normalized_targets
    
    def inverse_transform_targets(self, normalized_targets):
        """Convert normalized targets back to original scale."""
        return self.target_scaler.inverse_transform(normalized_targets.reshape(-1, 1)).flatten()

def preprocess_sequences(sequences):
    """Convert DNA sequences to integer encodings."""
    nuc_map = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}
    processed_seqs = []
    for seq in sequences:
        processed_seq = [nuc_map.get(nuc, 0) for nuc in seq.upper()]
        processed_seqs.append(processed_seq)
    return np.array(processed_seqs)

def load_and_preprocess_data(file_path, invert_targets=False):
    """Load and preprocess sgRNA data from file.
    
    Args:
        file_path: Path to CSV file containing sequences and scores
        invert_targets: If True, multiply targets by -1 (for KO data)
        
    Returns:
        tuple: Training and validation data splits and the fitted normalizer
    """
    data = pd.read_csv(file_path)
    data = data.dropna() 
    sequences = data['sequence'].values
    scores = data['score'].values
    
    # Convert sequences to numerical representation
    X = preprocess_sequences(sequences)
    y = scores
    
    # Invert targets if specified (for KO data)
    if invert_targets:
        y = -1 * y
    
    # Split data first
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit normalizer on training data only
    normalizer = DataNormalizer()
    X_train_norm, y_train_norm = normalizer.fit_transform(X_train, y_train)
    
    # Transform validation data using fitted normalizer
    X_val_norm, y_val_norm = normalizer.transform(X_val, y_val)
    
    print(f"\nNormalization stats for {file_path}:")
    print("Sequence mean:", normalizer.sequence_scaler.mean_)
    print("Sequence std:", np.sqrt(normalizer.sequence_scaler.var_))
    print("Target mean:", normalizer.target_scaler.mean_)
    print("Target std:", np.sqrt(normalizer.target_scaler.var_))
    
    return X_train_norm, X_val_norm, y_train_norm, y_val_norm, normalizer