import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import List, Union, Dict

from .models.base_model import BaseSequenceModel
from .data.preprocessing import preprocess_sequences

class SgRNAScorer:
    """Interface for sgRNA efficiency prediction.
    
    This class provides the main interface for other modules to get
    sgRNA efficiency predictions.
    """
    def __init__(self, model_path: str = None):
        """Initialize the scorer with a trained model.
        
        Args:
            model_path: Path to saved model weights. If None, uses default model.
        """
        if model_path is None:
            model_path = Path(__file__).parent / "weights" / "default_model.h5"
            
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path: Union[str, Path]) -> tf.keras.Model:
        """Load the trained model.
        
        Args:
            model_path: Path to model weights
            
        Returns:
            Loaded and compiled model
        """
        model = BaseSequenceModel(input_shape=(20,))  # Adjust shape as needed
        model.load_weights(str(model_path))
        return model
    
    def predict_sequence(self, sequence: str) -> float:
        """Predict efficiency score for a single sequence.
        
        Args:
            sequence: sgRNA sequence string
            
        Returns:
            Predicted efficiency score
        """
        processed_seq = preprocess_sequences([sequence])
        prediction = self.model.predict(processed_seq, verbose=0)
        return float(prediction[0][0])
    
    def predict_sequences(self, sequences: List[str]) -> np.ndarray:
        """Predict efficiency scores for multiple sequences.
        
        Args:
            sequences: List of sgRNA sequence strings
            
        Returns:
            Array of predicted efficiency scores
        """
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