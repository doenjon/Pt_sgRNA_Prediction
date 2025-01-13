import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import List, Union, Dict

from sgrna_scorer.data.preprocessing import preprocess_sequences
from sgrna_scorer.models.multi_path_atn_model import create_model

class SgRNAScorer:
    """Interface for sgRNA efficiency prediction."""
    
    def __init__(self, model_path=None):
        """Initialize the scorer with a trained model.
        
        Args:
            model_path: Path to saved model weights. If None, uses default model.
        """
        if model_path is None:
            # Look for the model trained in test.py
            model_path = Path(__file__).resolve().parent / "resources" / "model.weights.h5"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Default model weights not found at: {model_path}\n"
                    "Please specify a valid model_path or ensure the default model "
                    "is installed with the package."
                )
            
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path: Union[str, Path]) -> tf.keras.Model:
        """Load the trained model.
        
        Args:
            model_path: Path to model weights
            
        Returns:
            Loaded and compiled model
        
        Raises:
            FileNotFoundError: If model weights file doesn't exist
            ValueError: If weights file is invalid
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights not found at: {model_path}")
            
        try:
            model = create_model(input_shape=(20,))
            model.compile(optimizer='adam', loss='mse')  # Compile with basic settings
            model.load_weights(str(model_path))
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model weights: {str(e)}")
    
    def predict_sequence(self, sequence: str) -> float:
        """Predict efficiency score for a single sequence.
        
        Args:
            sequence: sgRNA sequence string (20 nt)
            
        Returns:
            Predicted efficiency score
            
        Raises:
            ValueError: If sequence length is not 20 nt or contains invalid characters
        """
        if len(sequence) != 20:
            raise ValueError("Sequence must be exactly 20 nucleotides long")
        if not set(sequence.upper()).issubset({'A', 'T', 'G', 'C', 'N'}):
            raise ValueError("Sequence contains invalid characters. Only A,T,G,C,N allowed")
            
        processed_seq = preprocess_sequences([sequence])
        prediction = self.model.predict(processed_seq, verbose=0)
        return float(prediction[0][0])
    
    def predict_sequences(self, sequences: List[str]) -> np.ndarray:
        """Predict efficiency scores for multiple sequences.
        
        Args:
            sequences: List of sgRNA sequence strings (20 nt each)
            
        Returns:
            Array of predicted efficiency scores
            
        Raises:
            ValueError: If any sequence is invalid
        """
        # Validate all sequences
        for seq in sequences:
            if len(seq) != 20:
                raise ValueError(f"All sequences must be exactly 20 nt long. Found: {seq}")
            if not set(seq.upper()).issubset({'A', 'T', 'G', 'C', 'N'}):
                raise ValueError(f"Invalid characters in sequence: {seq}")
                
        processed_seqs = preprocess_sequences(sequences)
        predictions = self.model.predict(processed_seqs, verbose=0)
        return predictions.flatten()
    
    def predict_sequences_with_metadata(
        self, sequences: List[str]
    ) -> List[Dict[str, Union[str, float]]]:
        """Predict efficiency scores and return detailed results.
        
        Args:
            sequences: List of sgRNA sequence strings
            
        Returns:
            List of dictionaries containing sequence and prediction details
        """
        scores = self.predict_sequences(sequences)
        
        results = []
        for sequence, score in zip(sequences, scores):
            results.append({
                'sequence': sequence,
                'efficiency_score': float(score),
                'sequence_length': len(sequence)
            })
        
        return results