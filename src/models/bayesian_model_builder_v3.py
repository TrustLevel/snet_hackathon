# src/models/bayesian_model_builder_v3.py 
# version 2 changed Stress_level
# version 3: modifed update_model +  build model
# version 4: include general initial trace
# version 5: added debug information
# version 6: build model update (gender_input) + update_model
# version 7: updated version to include example data and consistent variable handling
# version 7: simpliefied initiation - run successfully!
# version 8: build model (better integration of cpts)
# version 9: update generate initale trace (for faster initation)

import pymc as pm
import numpy as np
from typing import Dict, Optional
import logging

# Configure logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BayesianModelBuilder:
    """Builder for sleep quality Bayesian model with categorical modeling."""
    
    def __init__(self, concentration_factor: float = 1.0):
        """Initialize model builder with prior settings."""
        # Define valid categories
        self.categories = {
            'sleep_quality': ['Poor', 'Moderate', 'Good'],
            'gender': ['Male', 'Female'],
            'age': ['Young Adults', 'Middle-Aged', 'Older Adults'],
            'bmi': ['Underweight', 'Normal', 'Overweight', 'Obese'],
            'rhr': ['Low', 'Moderate', 'High'],
            'sleep_duration': ['Short', 'Moderate', 'Long'],
            'stress_level': ['Low', 'Moderate', 'High'],
            'blue_light': ['No Exposure', 'Low Exposure', 'Medium Exposure', 'High Exposure']
        }
        
        self.n_categories = len(self.categories['sleep_quality'])
        self.concentration_factor = concentration_factor
        
        # Initialize model state
        self.model = None
        self.trace = None
        
        logger.info("Initialized BayesianModelBuilder V3")
    
    def build_model(self) -> pm.Model:
        """Build PyMC model with CPT-based initialization."""
        
        # Import CPTs
        from src.models.cpts_v2 import (
            gender_sleep_quality_cpt,
            age_sleep_quality_cpt,
            bmi_sleep_quality_cpt,
            rhr_sleep_quality_cpt,
            sleep_duration_sleep_quality_cpt,
            stress_sleep_quality_cpt,
            blue_light_sleep_quality_cpt
        )
        
        # Create CPT mapping
        cpt_functions = {
            'gender': gender_sleep_quality_cpt,
            'age': age_sleep_quality_cpt,
            'bmi': bmi_sleep_quality_cpt,
            'rhr': rhr_sleep_quality_cpt,
            'sleep_duration': sleep_duration_sleep_quality_cpt,
            'stress_level': stress_sleep_quality_cpt,
            'blue_light': blue_light_sleep_quality_cpt
        }

        with pm.Model() as model:
            # Create input variables
            inputs = {}
            for category in self.categories.keys():
                if category != 'sleep_quality':
                    inputs[category] = pm.MutableData(
                        f'{category}_input',
                        np.zeros(1, dtype='int32')
                    )
                    logger.debug(f"Created input variable: {category}_input")

            # Create factor-specific effects
            effects = {}
            for category in self.categories.keys():
                if category != 'sleep_quality':
                    n_options = len(self.categories[category])
                    
                    # Get CPT for this category
                    cpt = cpt_functions[category]()
                    
                    # Convert CPT to weight matrix
                    weight_matrix = np.zeros((n_options, self.n_categories))
                    for i, cat in enumerate(self.categories[category]):
                        for j, quality in enumerate(self.categories['sleep_quality']):
                            weight_matrix[i, j] = cpt[cat][quality]
                    
                    # Scale by concentration factor
                    weight_matrix *= self.concentration_factor
                    
                    logger.debug(f"Initialized weights for {category}:")
                    logger.debug(f"Shape: {weight_matrix.shape}")
                    logger.debug(f"Values:\n{weight_matrix}")
                    
                    # Create Dirichlet weights using CPT probabilities
                    weights = pm.Dirichlet(
                        f'{category}_weights',
                        a=weight_matrix,
                        shape=(n_options, self.n_categories)
                    )
                    
                    # Select weights based on input
                    effects[category] = weights[inputs[category]]

            # Combine effects
            total_effect = sum(effects.values())
            
            # Convert to probabilities
            probs = pm.Deterministic(
                'category_probs',
                pm.math.softmax(total_effect)
            )
            
            # Setup sleep quality variable
            sleep_quality_data = pm.MutableData(
                'sleep_quality_data',
                np.zeros(1, dtype='int32')
            )
            
            # Create categorical outcome
            sleep_quality = pm.Categorical(
                'sleep_quality',
                p=probs,
                observed=sleep_quality_data
            )

            self.model = model
            
            # Generate initial trace
            self._generate_initial_trace()
            
            logger.info("Model built successfully")
            return model

    def _generate_initial_trace(self, n_samples: int = 100) -> None:
        """Generate initial trace by sampling from prior distributions."""
        if self.model is None:
            raise ValueError("Model must be built before generating trace")
            
        logger.info("Generating initial trace from priors...")
        logger.debug(f"Number of samples: {n_samples}")
        
        try:
            with self.model:
                self.trace = pm.sample(
                    draws=n_samples,
                    tune=n_samples // 2,
                    target_accept=0.9,
                    return_inferencedata=True,
                    chains=2,
                    progressbar=False
                )
                
                logger.info("Initial trace generated successfully")
                logger.debug(f"Variables in posterior: {list(self.trace.posterior.variables)}")
                
        except Exception as e:
            logger.error(f"Failed to generate initial trace: {str(e)}")
            raise

    def has_trace(self) -> bool:
        """Check if model has a valid trace."""
        return self.trace is not None

    def update_model(self, data: Dict[str, np.ndarray], n_samples: int = 1000) -> None:
        """Update model with new observations."""
        if self.model is None:
            raise ValueError("Model must be built before updating")
        
        try:
            with self.model:
                # Log current state
                logger.debug("Current model variables: %s", 
                            list(self.model.named_vars.keys()))
                logger.debug("Updating with data keys: %s", 
                            list(data.keys()))

                # Update input variables
                for category in self.categories.keys():
                    if category != 'sleep_quality':
                        var_name = f'{category}_input'
                        if var_name not in data:
                            logger.error(f"Missing required input data: {var_name}")
                            raise ValueError(f"Missing {var_name} in update data")
                            
                        logger.debug(f"Updating {var_name} shape={data[var_name].shape}")
                        pm.set_data({var_name: data[var_name]})
                
                # Update sleep quality data if provided
                if 'sleep_quality_data' in data:
                    pm.set_data({'sleep_quality_data': data['sleep_quality_data']})
                
                logger.info("Input data updated successfully")
                
                # Sample new trace
                logger.debug("Starting MCMC sampling...")
                self.trace = pm.sample(
                    draws=n_samples,
                    tune=n_samples // 2,
                    target_accept=0.9,
                    return_inferencedata=True,
                    chains=4,
                    progressbar=False
                )
                
                logger.info("Model updated successfully")
                
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to update model: {str(e)}")