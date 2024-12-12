import os
import sys
import numpy as np
from sgrna_scorer.models.multi_path_atn_model import create_model
from sgrna_scorer.utils.training import train_model, evaluate_model, plot_predictions, plot_training_history
from sgrna_scorer.data.preprocessing import load_and_preprocess_data
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ReduceLROnPlateau

# GPU settings
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.6.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=512)]
            )
        print("GPU memory limited to 512MB")
    except RuntimeError as e:
        print("Error setting GPU memory limit:", str(e))
        print("Falling back to CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    
    # Before training the model
    # Check if feature_train is not None before accessing its shape
    feature_train_shape = feature_train.shape if feature_train is not None else "N/A"
    feature_val_shape = feature_val.shape if feature_val is not None else "N/A"
    
    print(f"Training input shapes: X_train: {X_train.shape}, feature_train: {feature_train_shape}, y_train: {y_train.shape}")
    print(f"Validation input shapes: X_val: {X_val.shape}, feature_val: {feature_val_shape}, y_val: {y_val.shape}")
    
    # Ensure the model is expecting the correct input shapes
    print(f"Model input shapes: {model.input_shape}")
    
    # Train model
    if use_feature:
        train_data = (X_train, feature_train, y_train)
        val_data = (X_val, feature_val, y_val)
        
        history = train_model(
            model,
            train_data,
            val_data,
            use_feature=True,
            additional_callbacks=[reduce_lr],
            fold_num=fold_num
        )
        
        # Prepare feature inputs for evaluation as a list
        eval_val_data = [X_val, feature_val]
    else:
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        
        history = train_model(
            model,
            train_data,
            val_data,
            use_feature=False,
            additional_callbacks=[reduce_lr],
            fold_num=fold_num
        )
        
        # Prepare feature inputs for evaluation (exclude y_val)
        eval_val_data = X_val
    
    # Evaluate the model with correctly prepared feature inputs
    metrics = evaluate_model(model, eval_val_data, y_val)
    
    return metrics, history

def main():
    # Load and preprocess data
    data_file = 'sgrna_scorer/resources/pt_sat_guides.pos.clean.csv'  
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
    
    print("\nModel with position feature:")
    avg_mse_with_feature = np.mean([m['mse'] for m in metrics_with_feature])
    avg_mae_with_feature = np.mean([m['mae'] for m in metrics_with_feature])
    avg_corr_with_feature = np.mean([m['spearman_corr'] for m in metrics_with_feature])
    print(f"Average MSE: {avg_mse_with_feature:.4f} (±{np.std([m['mse'] for m in metrics_with_feature]):.4f})")

if __name__ == "__main__":
    main()