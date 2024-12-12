import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class DataNormalizer:
    def __init__(self):
        self.target_scaler = StandardScaler()
    
    def plot_distributions(self, original_data, normalized_data, data_type="targets"):
        """Plot distributions before and after normalization."""
        plt.figure(figsize=(12, 5))
        
        # Before normalization
        plt.subplot(1, 2, 1)
        plt.hist(original_data.flatten(), bins=50, alpha=0.7)
        plt.title(f'Original {data_type} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Count')
        
        # After normalization
        plt.subplot(1, 2, 2)
        plt.hist(normalized_data.flatten(), bins=50, alpha=0.7)
        plt.title(f'Normalized {data_type} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'logs/normalization_{data_type}.png')
        plt.close()
        
        # Print statistics
        print(f"\n{data_type} Statistics:")
        print("Before normalization:")
        print(f"Mean: {np.mean(original_data):.3f}")
        print(f"Std: {np.std(original_data):.3f}")
        print(f"Min: {np.min(original_data):.3f}")
        print(f"Max: {np.max(original_data):.3f}")
        
        print("\nAfter normalization:")
        print(f"Mean: {np.mean(normalized_data):.3f}")
        print(f"Std: {np.std(normalized_data):.3f}")
        print(f"Min: {np.min(normalized_data):.3f}")
        print(f"Max: {np.max(normalized_data):.3f}")
        
    def fit_transform(self, sequences, targets):
        """Fit and transform targets only, leave sequences as is."""
        # Store original targets for plotting
        original_targets = targets.copy()
        
        # Fit and transform targets
        normalized_targets = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # Plot distributions for targets only
        self.plot_distributions(original_targets, normalized_targets, "targets")
        
        return sequences, normalized_targets
    
    def transform(self, sequences, targets):
        """Transform targets only, leave sequences as is."""
        # Transform targets
        normalized_targets = self.target_scaler.transform(targets.reshape(-1, 1)).flatten()
        
        return sequences, normalized_targets
    
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
    """Load and preprocess sgRNA data from file."""
    data = pd.read_csv(file_path)
    data = data.dropna() 
    sequences = data['sequence'].values
    scores = data['score'].values
    positions = data['position'].values  # Load position feature
    
    # Convert sequences to numerical representation
    X = preprocess_sequences(sequences)
    y = scores
    
    # Add position feature as an additional column to X
    X = np.column_stack([X, positions])
    
    # Store original data for comparison
    X_original = X.copy()
    y_original = y.copy()
    
    # Invert targets if specified (for KO data) - do this BEFORE splitting
    if invert_targets:
        y = -1 * y
    
    # Split data after inversion
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit normalizer on training data only
    normalizer = DataNormalizer()
    X_train_norm, y_train_norm = normalizer.fit_transform(X_train, y_train)
    
    # Transform validation data using fitted normalizer
    X_val_norm, y_val_norm = normalizer.transform(X_val, y_val)
    
    # Return both normalized and original data for comparison
    return (X_train_norm, X_val_norm, y_train_norm, y_val_norm, normalizer,
            X_original, y_original)