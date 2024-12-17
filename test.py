import os
import sys
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ml cuda/12.2 cudnn/8.9


os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.6.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

from sgrna_scorer.data.preprocessing import load_and_preprocess_data
from sgrna_scorer.models.multi_path_atn_model import create_model

def train_and_evaluate(X_train, X_val, y_train, y_val, feature_train=None, feature_val=None, use_feature=False, fold_num=0):
    """Train and evaluate model for one fold."""
    sequence_length = X_train.shape[1]
    
    # Create and compile model
    model = create_model(sequence_input_shape=(sequence_length,), use_feature=use_feature)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=0
    )
    
    # Train model
    if use_feature:
        history = model.fit(
            [X_train, feature_train],
            y_train,
            validation_data=([X_val, feature_val], y_val),
            epochs=100,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                reduce_lr
            ],
            verbose=1
        )
        
        # Evaluate
        val_loss, val_mae = model.evaluate([X_val, feature_val], y_val, verbose=0)
        y_pred = model.predict([X_val, feature_val], verbose=0)
    else:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                reduce_lr
            ],
            verbose=1
        )
        
        # Evaluate
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        y_pred = model.predict(X_val, verbose=0)
    
    # Calculate additional metrics
    spearman_corr = spearmanr(y_val, y_pred.flatten())[0]
    metrics = {
        'mse': val_loss,
        'mae': val_mae,
        'spearman_corr': spearman_corr
    }
    
    return metrics, history

def preprocess_sequences(sequences, target_length, pad_front=True):
    """Convert DNA sequences to integer encodings and adjust length."""
    nuc_map = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}
    processed_seqs = []
    for seq in sequences:
        processed_seq = [nuc_map.get(nuc, 0) for nuc in seq.upper()]
        if len(processed_seq) < target_length:
            padding = [0] * (target_length - len(processed_seq))
            if pad_front:
                processed_seq = padding + processed_seq
            else:
                processed_seq = processed_seq + padding
        else:
            if pad_front:
                processed_seq = processed_seq[-target_length:]
            else:
                processed_seq = processed_seq[:target_length]
        processed_seqs.append(processed_seq)
    return np.array(processed_seqs)

def main():
    # Load and preprocess data
    data_file = 'sgrna_scorer/resources/pt_sat_guides.60.clean.csv'
    X_train, X_val, y_train, y_val, normalizer, X_original, y_original = load_and_preprocess_data(
        data_file, 
        invert_targets=False
    )
    
    # Define range of sequence lengths to test
    sequence_lengths = range(20, 31)  # Test from 20 to 30 base pairs
    
    # Store results for each sequence length
    results = {}
    
    for seq_length in sequence_lengths:
        print(f"\n=== Testing with sequence length: {seq_length} ===")
        
        # Preprocess sequences to the desired length
        X_train_adjusted = preprocess_sequences(X_train, seq_length, pad_front=True)
        X_val_adjusted = preprocess_sequences(X_val, seq_length, pad_front=True)
        
        # Train and evaluate model
        metrics, _ = train_and_evaluate(
            X_train_adjusted, X_val_adjusted,
            y_train, y_val,
            use_feature=False
        )
        
        # Store results
        results[seq_length] = metrics
    
    # Print results
    for seq_length, metrics in results.items():
        print(f"\nResults for sequence length {seq_length}:")
        print(f"MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, Spearman Corr: {metrics['spearman_corr']:.4f}")

if __name__ == "__main__":
    main()