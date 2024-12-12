import os
import sys
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

def main():
    # Load and preprocess data
    data_file = 'sgrna_scorer/resources/pt_sat_guides.fea.clean.csv'  
    X_train, X_val, y_train, y_val, normalizer, X_original, y_original = load_and_preprocess_data(
        data_file, 
        invert_targets=False
    )
    
    # Combine train and validation sets for cross-validation
    X = np.concatenate([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    
    # Extract only the position feature (assuming it's the last column)
    position_feature = X[:, -1].reshape(-1, 1)  # Reshape to be 2D
    # Remove feature columns from X
    X = X[:, :-1]  # Remove the last column which is the position feature
    
    # Initialize K-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    metrics_no_feature = []
    metrics_with_feature = []
    
    # Perform cross-validation
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold_num + 1}/5 ===")
        
        # Split data for this fold
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        feature_train_fold = position_feature[train_idx]  # Use only position feature
        feature_val_fold = position_feature[val_idx]      # Use only position feature
        
        # Train and evaluate model without feature
        print("\n--- Training Model Without Feature ---")
        fold_metrics_no_feature, history_no_feature = train_and_evaluate(
            X_train_fold, X_val_fold,
            y_train_fold, y_val_fold,
            use_feature=False,
            fold_num=fold_num
        )
        metrics_no_feature.append(fold_metrics_no_feature)
        
        # Train and evaluate model with position feature
        print("\n--- Training Model With Position Feature ---")
        fold_metrics_with_feature, history_with_feature = train_and_evaluate(
            X_train_fold, X_val_fold,
            y_train_fold, y_val_fold,
            feature_train_fold, feature_val_fold,
            use_feature=True,
            fold_num=fold_num
        )
        metrics_with_feature.append(fold_metrics_with_feature)
    
    # Calculate and print average results
    print("\n=== Average Results Across 5 Folds ===")
    
    print("\nModel without feature:")
    avg_mse_no_feature = np.mean([m['mse'] for m in metrics_no_feature])
    avg_mae_no_feature = np.mean([m['mae'] for m in metrics_no_feature])
    avg_corr_no_feature = np.mean([m['spearman_corr'] for m in metrics_no_feature])
    print(f"Average MSE: {avg_mse_no_feature:.4f} (±{np.std([m['mse'] for m in metrics_no_feature]):.4f})")
    print(f"Average MAE: {avg_mae_no_feature:.4f} (±{np.std([m['mae'] for m in metrics_no_feature]):.4f})")
    print(f"Average Spearman correlation: {avg_corr_no_feature:.4f} (±{np.std([m['spearman_corr'] for m in metrics_no_feature]):.4f})")
    
    print("\nModel with position feature:")
    avg_mse_with_feature = np.mean([m['mse'] for m in metrics_with_feature])
    avg_mae_with_feature = np.mean([m['mae'] for m in metrics_with_feature])
    avg_corr_with_feature = np.mean([m['spearman_corr'] for m in metrics_with_feature])
    print(f"Average MSE: {avg_mse_with_feature:.4f} (±{np.std([m['mse'] for m in metrics_with_feature]):.4f})")
    print(f"Average MAE: {avg_mae_with_feature:.4f} (±{np.std([m['mae'] for m in metrics_with_feature]):.4f})")
    print(f"Average Spearman correlation: {avg_corr_with_feature:.4f} (±{np.std([m['spearman_corr'] for m in metrics_with_feature]):.4f})")

    # Calculate percent improvement
    mse_improvement = ((avg_mse_no_feature - avg_mse_with_feature) / avg_mse_no_feature) * 100
    mae_improvement = ((avg_mae_no_feature - avg_mae_with_feature) / avg_mae_no_feature) * 100
    corr_improvement = ((avg_corr_with_feature - avg_corr_no_feature) / avg_corr_no_feature) * 100 if avg_corr_no_feature != 0 else float('inf')

    print(f"\nPercent Improvement in MSE: {mse_improvement:.2f}%")
    print(f"Percent Improvement in MAE: {mae_improvement:.2f}%")
    print(f"Percent Improvement in Spearman correlation: {corr_improvement:.2f}%")

if __name__ == "__main__":
    main()