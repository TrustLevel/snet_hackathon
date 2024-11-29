# V2 changes: 
# - Adaption to new SQP file (update store_prediction method)
# - Added record_feedback method
# - Added method to store sleep quality prediction category
# V3 changes: add calibration process into get_latest_evaluation_results method, added "union"
# tested 26.11 with app_v5 

import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Optional, Any, Union
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SleepDataStorage:
    """CSV-based storage for sleep quality predictions and evaluations."""
    
    PREDICTIONS_SCHEMA = {
        'prediction_id': 'str',
        'timestamp': 'str',
        'prob_poor': 'float64',
        'prob_moderate': 'float64',
        'prob_good': 'float64',
        'predicted_category': 'str',
        'actual_quality': 'float64',
        'actual_category': 'str',
        'feedback_timestamp': 'str',
        'input_data': 'str',
        'calibrated_prob_poor': 'float64',
        'calibrated_prob_moderate': 'float64',
        'calibrated_prob_good': 'float64',
        'insights': 'str',
        'risk_metrics': 'str'
    }
    
    EVALUATIONS_SCHEMA = {
        'batch_id': 'str',
        'timestamp': 'str',
        'prediction_ids': 'str',  # JSON list
        'accuracy': 'float64',
        'decisiveness': 'float64',
        'robustness': 'float64',
        'num_predictions': 'int64'
    }
    
    def __init__(self, storage_dir: str = "src/data"):
        """Initialize storage with directory path."""
        self.storage_dir = storage_dir
        self.predictions_file = os.path.join(storage_dir, "predictions.csv")
        self.evaluations_file = os.path.join(storage_dir, "evaluation_history.csv")
        
        # Create directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize files if they don't exist
        self._init_predictions_file()
        self._init_evaluations_file()
    
    def _create_empty_dataframe(self, schema: Dict[str, str]) -> pd.DataFrame:
        """Create empty DataFrame with specified schema."""
        # Initialize with empty values matching types
        data = {col: [] for col in schema.keys()}
        df = pd.DataFrame(data)
        
        # Set types for numeric columns only
        for col, dtype in schema.items():
            if 'float' in dtype or 'int' in dtype:
                df[col] = df[col].astype(dtype)
        
        return df
    
    def _init_predictions_file(self):
        """Create predictions.csv if it doesn't exist."""
        if not os.path.exists(self.predictions_file):
            df = self._create_empty_dataframe(self.PREDICTIONS_SCHEMA)
            df.to_csv(self.predictions_file, index=False)
            logger.info(f"Created predictions file: {self.predictions_file}")
    
    def _init_evaluations_file(self):
        """Create evaluation_history.csv if it doesn't exist."""
        if not os.path.exists(self.evaluations_file):
            df = self._create_empty_dataframe(self.EVALUATIONS_SCHEMA)
            df.to_csv(self.evaluations_file, index=False)
            logger.info(f"Created evaluations file: {self.evaluations_file}")

    def store_prediction(self,
                        prediction_id: str,
                        probabilities: Dict[str, float],
                        predicted_category: str,
                        input_data: Dict[str, Any],
                        insights: Dict[str, Any],
                        calibrated_probabilities: Optional[Dict[str, float]] = None) -> str:
        """Store a new prediction with all associated data."""
        try:
            # Create new data matching ALL schema fields
            new_data = {
                'prediction_id': [prediction_id],
                'timestamp': [datetime.now().isoformat()],
                'prob_poor': [float(probabilities['Poor'])],
                'prob_moderate': [float(probabilities['Moderate'])],
                'prob_good': [float(probabilities['Good'])],
                'predicted_category': [str(predicted_category)],
                'actual_quality': [np.nan],  # Use np.nan for numeric nulls
                'actual_category': [None],  # Initialize as None
                'feedback_timestamp': [None],
                'input_data': [json.dumps(input_data)],
                'calibrated_prob_poor': [float(calibrated_probabilities['Poor']) if calibrated_probabilities else np.nan],
                'calibrated_prob_moderate': [float(calibrated_probabilities['Moderate']) if calibrated_probabilities else np.nan],
                'calibrated_prob_good': [float(calibrated_probabilities['Good']) if calibrated_probabilities else np.nan],
                'insights': [json.dumps(insights)],
                'risk_metrics': [None]
            }
            
            # Create new row
            new_row = pd.DataFrame(new_data)
            
            if os.path.exists(self.predictions_file):
                df = pd.read_csv(self.predictions_file)
                # Add any missing columns from schema
                for col in self.PREDICTIONS_SCHEMA.keys():
                    if col not in df.columns:
                        df[col] = None
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                df = new_row

            # Set dtypes after concatenation for numeric columns only
            for col, dtype in self.PREDICTIONS_SCHEMA.items():
                if 'float' in dtype or 'int' in dtype:
                    df[col] = df[col].astype(dtype)
            
            df.to_csv(self.predictions_file, index=False)
            logger.info(f"Stored prediction: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            raise RuntimeError(f"Failed to store prediction: {str(e)}")

    @staticmethod
    def _score_to_category(score: float) -> str:
        """Convert numeric sleep quality score to category."""
        if pd.isna(score):
            return None
        if score <= 5:
            return "Poor"
        elif score <= 7:
            return "Moderate"
        else:
            return "Good"

    def record_feedback(self,
                       prediction_id: str,
                       actual_quality: float) -> bool:
        """Record actual sleep quality feedback."""
        try:
            df = pd.read_csv(self.predictions_file)
            
            # Find prediction
            mask = df['prediction_id'] == prediction_id
            if not any(mask):
                logger.warning(f"Prediction {prediction_id} not found")
                return False
            
            # Update with correct types
            current_timestamp = datetime.now().isoformat()
            quality_float = float(actual_quality)
            
            df.loc[mask, 'actual_quality'] = quality_float
            df.loc[mask, 'actual_category'] = self._score_to_category(quality_float)
            df.loc[mask, 'feedback_timestamp'] = current_timestamp
            
            # Set dtypes for numeric columns only
            for col, dtype in self.PREDICTIONS_SCHEMA.items():
                if 'float' in dtype or 'int' in dtype:
                    df[col] = df[col].astype(dtype)
            
            df.to_csv(self.predictions_file, index=False)
            logger.info(f"Recorded feedback for prediction: {prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
            return False

    def get_predictions_with_outcomes(self, 
                                    limit: Optional[int] = None,
                                    min_predictions: int = 3) -> List[Dict[str, Any]]:
        """Get predictions that have feedback for evaluation or display."""
        try:
            df = pd.read_csv(self.predictions_file)
            
            # Get only predictions with feedback
            df = df.dropna(subset=['actual_quality'])
            
            if len(df) < min_predictions:
                logger.info(f"Not enough predictions with feedback: {len(df)}")
                return []
            
            if limit:
                df = df.tail(limit)
            
            # Convert rows to dictionaries
            predictions = []
            for _, row in df.iterrows():
                pred = {
                    'prediction_id': row['prediction_id'],
                    'probabilities': {
                        'Poor': float(row['prob_poor']),
                        'Moderate': float(row['prob_moderate']),
                        'Good': float(row['prob_good'])
                    },
                    'actual_quality': float(row['actual_quality']),
                    'actual_category': self._score_to_category(float(row['actual_quality'])),
                    'timestamp': row['timestamp'],
                    'feedback_timestamp': row['feedback_timestamp'],
                    'predicted_category': row['predicted_category'],
                    'insights': json.loads(row['insights']) if pd.notna(row['insights']) else None
                }
                predictions.append(pred)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            return []

    def get_predictions_for_evaluation(self, 
                                     min_predictions: int = 10,
                                     max_predictions: int = 100) -> Optional[Dict[str, Any]]:
        """Get batch of predictions with feedback for SNET evaluation."""
        try:
            df = pd.read_csv(self.predictions_file)
            
            # Get predictions with feedback
            mask = df['actual_quality'].notna()
            df_with_feedback = df[mask].copy()
            
            if len(df_with_feedback) < min_predictions:
                logger.info(f"Insufficient predictions with feedback: {len(df_with_feedback)}")
                return None
            
            # Take up to max_predictions most recent entries
            batch = df_with_feedback.tail(max_predictions)
            
            predictions = []
            for _, row in batch.iterrows():
                pred = {
                    'probabilities': {
                        'Poor': float(row['prob_poor']),
                        'Moderate': float(row['prob_moderate']),
                        'Good': float(row['prob_good'])
                    },
                    'actual_category': self._score_to_category(float(row['actual_quality']))
                }
                predictions.append(pred)
            
            batch_id = f"batch_{datetime.now().isoformat()}"
            
            return {
                'batch_id': batch_id,
                'predictions': predictions,
                'prediction_ids': batch['prediction_id'].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error getting predictions for evaluation: {str(e)}")
            return None

    def store_evaluation_results(self, 
                               batch_id: str,
                               prediction_ids: List[str],
                               metrics: Dict[str, float]) -> bool:
        """Store evaluation results from SNET service."""
        try:
            new_data = {
                'batch_id': [batch_id],
                'timestamp': [datetime.now().isoformat()],
                'prediction_ids': [json.dumps(prediction_ids)],
                'accuracy': [float(metrics['accuracy'])],
                'decisiveness': [float(metrics['decisiveness'])],
                'robustness': [float(metrics['robustness'])],
                'num_predictions': [len(prediction_ids)]
            }
            
            # Create new row
            new_row = pd.DataFrame(new_data)
            
            if os.path.exists(self.evaluations_file):
                df = pd.read_csv(self.evaluations_file)
                # Add any missing columns
                for col in self.EVALUATIONS_SCHEMA.keys():
                    if col not in df.columns:
                        df[col] = None
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                df = new_row
            
            # Set dtypes for numeric columns only
            for col, dtype in self.EVALUATIONS_SCHEMA.items():
                if 'float' in dtype or 'int' in dtype:
                    df[col] = df[col].astype(dtype)
            
            df.to_csv(self.evaluations_file, index=False)
            logger.info(f"Stored evaluation results for batch: {batch_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing evaluation results: {str(e)}")
            return False

    def get_latest_evaluation_results(self, limit: int = 3, metrics_only: bool = False) -> Optional[Union[pd.DataFrame, Dict[str, float]]]:
        """
        Get most recent evaluation results.
        
        Args:
            limit: Number of results to return
            metrics_only: If True, returns only ADR metrics from most recent evaluation
            
        Returns:
            DataFrame of recent results or dict of latest ADR metrics if metrics_only=True
        """
        try:
            if not os.path.exists(self.evaluations_file):
                return None
                
            df = pd.read_csv(self.evaluations_file)
            if df.empty:
                return None
                
            if metrics_only:
                latest = df.iloc[-1]
                return {
                    'accuracy': float(latest['accuracy']),
                    'decisiveness': float(latest['decisiveness']),
                    'robustness': float(latest['robustness'])
                }
                
            return df.tail(limit)
                
        except Exception as e:
            logger.error(f"Error getting evaluation results: {str(e)}")
            return None

    def clear_test_data(self):
        """Clear all data (use only in testing)."""
        if 'test_data' in self.storage_dir:  # Safety check
            # Create empty DataFrames with correct schemas
            pred_df = self._create_empty_dataframe(self.PREDICTIONS_SCHEMA)
            eval_df = self._create_empty_dataframe(self.EVALUATIONS_SCHEMA)
            
            # Save empty files
            pred_df.to_csv(self.predictions_file, index=False)
            eval_df.to_csv(self.evaluations_file, index=False)
            logger.info("Cleared test data")
