#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from src.utils.storage_v3 import SleepDataStorage
from src.integrations.snet_integration_v3 import SNETIntegration
from src.models.bayesian_model_builder_v3 import BayesianModelBuilder
from src.models.calibration_manager import CalibrationManager

def run_calibration():
    """Run the calibration process."""
    print("\n=== Running Model Calibration ===")
    
    # Initialize components
    storage = SleepDataStorage()
    builder = BayesianModelBuilder(concentration_factor=1.0)
    snet = SNETIntegration(
        private_key=os.getenv('SNET_PRIVATE_KEY'),
        eth_rpc_endpoint=os.getenv('SNET_ETH_ENDPOINT')
    )
    
    # Create calibration manager
    calibration = CalibrationManager(builder, storage, snet)
    
    # Get current predictions count
    predictions = storage.get_predictions_with_outcomes()
    print(f"\nFound {len(predictions)} predictions with feedback in system")
    print(f"Using last 10 predictions for calibration")
    
    print("\nRunning Photrek Risk Assessment...")
    result = calibration.run_calibration()
    
    print("\nCalibration Results:")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print("\nADR Metrics:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.3f}")
            
        print("\nModel Calibration:")
        cf = result['concentration_factor']
        print(f"  Previous concentration: {cf['previous']:.3f}")
        print(f"  New concentration: {cf['new']:.3f}")
        print(f"  Model updated: {'Yes' if cf['updated'] else 'No'}")
        
        if result.get('insights'):
            print("\nInsights:")
            insights = result['insights']
            if insights.get('warnings'):
                print("\nWarnings:")
                for warning in insights['warnings']:
                    print(f"  ! {warning}")
            if insights.get('recommendations'):
                print("\nRecommendations:")
                for rec in insights['recommendations']:
                    print(f"  > {rec}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    run_calibration()