# src/models/sleep_quality_predict_v5.py
# tested: aligned with bayesian model builder v3
# version 2: add logging to prepare data for update for bugfix 

from typing import Dict, Optional, Any, List
import numpy as np
from datetime import datetime
import pymc as pm
import json 
from src.utils.preprocessing import preprocess_user_data
from src.utils.storage_v3 import SleepDataStorage
from src.models.bayesian_model_builder_v4 import BayesianModelBuilder

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SleepQualityPredict:
    """
    Manages sleep quality predictions using a persistent Bayesian model.
    Uses example data for initialization and learns from feedback.
    """
    
    def __init__(self, storage_dir: str = "src/data"):
        """Initialize sleep quality prediction system."""
        # Initialize core components
        self.builder = BayesianModelBuilder()
        self.storage = SleepDataStorage(storage_dir)
        
        # Initialize model
        self._initialize_model()
        logger.info("Initialized SleepQualityPredict")
        
    def _initialize_model(self) -> None:
        """Initialize Bayesian model"""
        logger.info("Initializing Bayesian model...")
        
        # Build initial model only
        self.builder.build_model()
        logger.info("Model initialized successfully")
       
    def _prepare_update_data(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Convert feedback data to model update format."""
        logger.debug(f"_prepare_update_data called with {len(feedback_data)} samples")

        # Take only last 10 samples
        feedback_data = feedback_data[-10:]
        logger.info(f"Using last {len(feedback_data)} samples for update")

        # Filter for valid samples
        valid_samples = []
        for sample in feedback_data:
            try:
                # Check for required fields
                if not sample.get('input_data') or not sample.get('actual_category'):
                    continue
                    
                # Try to parse input_data
                input_data = json.loads(sample['input_data'])
                if all(category in input_data for category in self.builder.categories.keys() if category != 'sleep_quality'):
                    valid_samples.append(sample)
                
            except json.JSONDecodeError:
                continue

        if not valid_samples:
            logger.warning("No valid samples found after validation")
            return None
                
        n_samples = len(valid_samples)
        logger.info(f"Found {n_samples} valid samples for update")
        
        # Initialize arrays
        update_data = {}
        for category in self.builder.categories.keys():
            if category != 'sleep_quality':
                var_name = f'{category}_input'
                update_data[var_name] = np.zeros(n_samples, dtype='int32')
        
        update_data['sleep_quality_data'] = np.zeros(n_samples, dtype='int32')
        
        # Fill arrays with data
        for i, sample in enumerate(valid_samples):
            input_data = json.loads(sample['input_data'])
            
            # Convert input categories to indices
            for category in self.builder.categories.keys():
                if category != 'sleep_quality':
                    var_name = f'{category}_input'
                    value = input_data[category]
                    index = self.builder.categories[category].index(value)
                    update_data[var_name][i] = index
            
            # Add outcome data
            quality_category = sample['actual_category']
            update_data['sleep_quality_data'][i] = \
                self.builder.categories['sleep_quality'].index(quality_category)
        
        return update_data

    def predict(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a sleep quality prediction."""
        try:
            logger.info("Starting new prediction...")
            
            # Preprocess input data
            processed_data = preprocess_user_data(raw_data)
            logger.debug(f"Processed data: {processed_data}")
            
            # Generate prediction ID
            prediction_id = f"pred_{datetime.now().isoformat()}"
            
            # Convert input to model format
            model_input = self._prepare_prediction_input(processed_data)
            logger.debug("Prepared model input")
            
            # Generate prediction
            with self.builder.model:
                # Set input data
                for name, value in model_input.items():
                    pm.set_data({name: value})
                
                # Sample from posterior predictive
                predictions = pm.sample_posterior_predictive(
                    self.builder.trace,
                    var_names=['category_probs']
                )
                
                # Calculate probabilities
                probs = predictions.posterior_predictive['category_probs'].mean(
                    dim=('chain', 'draw')
                ).values[0]
                
                # Convert to dictionary
                probabilities = {
                    category: float(prob)
                    for category, prob in zip(self.builder.categories['sleep_quality'], probs)
                }
                
                # Get predicted category
                predicted_category = max(probabilities.items(), key=lambda x: x[1])[0]
            
            # Store prediction
            self.storage.store_prediction(
                prediction_id=prediction_id,
                probabilities=probabilities,
                predicted_category=predicted_category,
                input_data=processed_data,
                insights=self._generate_insights(probabilities, processed_data)
            )
            
            result = {
                'prediction_id': prediction_id,
                'probabilities': probabilities,
                'predicted_category': predicted_category,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction completed: {prediction_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def _prepare_prediction_input(self, processed_data: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Convert processed input data to model format."""
        input_data = {}
        
        for category in self.builder.categories.keys():
            if category != 'sleep_quality':
                var_name = f'{category}_input'
                value = processed_data[category]
                index = self.builder.categories[category].index(value)
                input_data[var_name] = np.array([index], dtype='int32')
        
        return input_data

    def record_actual_quality(self, prediction_id: str, actual_quality: float) -> Dict[str, Any]:
        """Record actual sleep quality feedback."""
        try:
            success = self.storage.record_feedback(
                prediction_id=prediction_id,
                actual_quality=actual_quality
            )
            
            logger.debug(f"Storage feedback result: {success}") 

            if not success:
                raise ValueError(f"Failed to record feedback for prediction {prediction_id}")
                    
            result = {
                'prediction_id': prediction_id,
                'actual_quality': actual_quality,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }

            logger.debug("Completed record_actual_quality successfully")  # Add this
            return result

        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
            raise

    def update_model_with_feedback(self) -> Dict[str, Any]:
        """
        Update model with latest feedback. Called when update button is pressed.
        Returns status dictionary for app feedback.
        """
        try:
            # Get all feedback samples
            feedback_data = self.storage.get_predictions_with_outcomes()
            if not feedback_data:
                logger.info("No feedback data available")
                return {
                    'status': 'error',
                    'message': 'No feedback data available',
                    'timestamp': datetime.now().isoformat()
                }
                
            # Take last 10 samples
            feedback_data = feedback_data[-10:]
            logger.info(f"Processing last {len(feedback_data)} feedback samples")
            
            # Log most recent feedback for debugging
            if feedback_data:
                logger.info(f"Most recent feedback: {feedback_data[-1]}")
            
            # Prepare data
            update_data = self._prepare_update_data(feedback_data)
            
            if update_data is None:
                logger.info("No valid data for update")
                return {
                    'status': 'error',
                    'message': 'No valid samples for update',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Add shape logging
            logger.debug(f"Update data shapes: {[f'{k}: {v.shape}' for k, v in update_data.items()]}")
                
            # Update model
            self.builder.update_model(update_data)
            logger.info("Model updated successfully with feedback")
            
            return {
                'status': 'success',
                'samples_used': len(feedback_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _generate_insights(self, 
                         probabilities: Dict[str, float],
                         processed_data: Dict[str, str]) -> Dict[str, Any]:
        """Generate insights from prediction."""
        return {
            'confidence': max(probabilities.values()),
            'key_factors': [
                {
                    'factor': factor,
                    'category': value,
                    'importance': 0.15 if factor in ['bmi', 'rhr'] else 0.25 if factor == 'sleep_duration' else 0.3
                }
                for factor, value in processed_data.items()
                if factor != 'gender'  # Exclude gender from key factors
            ]
        }