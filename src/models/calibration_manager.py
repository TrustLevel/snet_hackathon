from typing import Dict, Any, List
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalibrationManager:
   """Manages evaluation and calibration of the sleep quality prediction model."""
   
   def __init__(self, 
                builder, 
                storage,
                snet_integration,
                min_samples: int = 10):
       """Initialize calibration manager."""
       self.builder = builder
       self.storage = storage
       self.snet_integration = snet_integration
       self.min_samples = min_samples
       
       # Category mapping (storage format to indices)
       self.category_to_index = {
           'Poor': 0,
           'Moderate': 1, 
           'Good': 2
       }

   def run_calibration(self) -> Dict[str, Any]:
       """
       Run complete calibration cycle.
       Uses only the 10 most recent predictions for demonstration purposes.
       """
       try:
           # Get all predictions with outcomes
           all_predictions = self.storage.get_predictions_with_outcomes()
           
           # Take only the last 10 predictions for demo purposes
           predictions = all_predictions[-10:] if len(all_predictions) >= 10 else all_predictions
           
           if len(predictions) < self.min_samples:
               logger.info(f"Insufficient samples: {len(predictions)} < {self.min_samples}")
               return {
                   "status": "insufficient_data",
                   "message": f"Need at least {self.min_samples} samples with feedback",
                   "current_samples": len(predictions)
               }
           
           logger.info(f"Running calibration with {len(predictions)} most recent predictions")
           
           # Format predictions for SNET evaluation
           prediction_vectors = []
           actuals = []
           
           for pred in predictions:
               # Create probability vector [Poor, Moderate, Good]
               prob_vector = [
                   float(pred['probabilities']['Poor']),
                   float(pred['probabilities']['Moderate']),
                   float(pred['probabilities']['Good'])
               ]
               prediction_vectors.append(prob_vector)
               
               # Convert category to index (0-based)
               actuals.append(self.category_to_index[pred['actual_category']])
           
           # Get ADR metrics from Photrek
           try:
               adr_metrics = self.snet_integration.evaluate_predictions(prediction_vectors, actuals)
               logger.info(f"Received ADR metrics: {adr_metrics}")
           except Exception as e:
               logger.error(f"Error getting ADR metrics: {str(e)}")
               return {"status": "error", "message": f"ADR evaluation failed: {str(e)}"}

           # Determine if calibration is needed
           current_factor = self.builder.concentration_factor
           new_factor = self._calculate_new_factor(adr_metrics)
           needs_update = abs(new_factor - current_factor) > 0.001  # Small threshold for changes

           # Update builder if needed
           if needs_update:
               logger.info(f"Updating concentration factor: {current_factor} -> {new_factor}")
               self.builder.concentration_factor = new_factor
               
           # Format result
           result = {
               'status': 'success',
               'timestamp': datetime.now().isoformat(),
               'metrics': {
                   'accuracy': float(adr_metrics['accuracy']),
                   'decisiveness': float(adr_metrics['decisiveness']),
                   'robustness': float(adr_metrics['robustness'])
               },
               'concentration_factor': {
                   'previous': float(current_factor),
                   'new': float(new_factor),
                   'updated': needs_update
               },
               'samples_evaluated': len(predictions),
               'insights': self._generate_insights(adr_metrics)
           }
           
           return result
           
       except Exception as e:
           logger.error(f"Calibration failed: {str(e)}")
           return {"status": "error", "message": str(e)}

   def _calculate_new_factor(self, metrics: Dict[str, float]) -> float:
       """Calculate new concentration factor based on ADR metrics."""
       current = self.builder.concentration_factor
       accuracy = metrics['accuracy']
       
       if accuracy < 0.7:
           # Model is overconfident - reduce concentration
           return current * 0.8
       elif accuracy > 0.9:
           # Model is underconfident - increase concentration
           return current * 1.2
       
       return current

   def _generate_insights(self, metrics: Dict[str, float]) -> Dict[str, Any]:
       """Generate insights from ADR metrics."""
       warnings = []
       recommendations = []
       
       if metrics['accuracy'] < 0.7:
           warnings.append("Model accuracy below threshold")
           recommendations.append("Consider gathering more diverse feedback")
       
       if metrics['decisiveness'] < 0.6:
           warnings.append("Low decisiveness indicates uncertain predictions")
           recommendations.append("Model may need stronger priors")
           
       if metrics['robustness'] < 0.5:
           warnings.append("Low robustness suggests poor handling of edge cases")
           recommendations.append("Review predictions with low probabilities")
       
       return {
           'warnings': warnings,
           'recommendations': recommendations
       }