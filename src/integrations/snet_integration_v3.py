# File: src/integrations/snet_integration_v3.py

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import base64
import grpc
import io
import numpy as np
import pandas as pd
from snet.sdk import SnetSDK
from snet.sdk.config import Config
from web3.exceptions import ContractLogicError
from src.protos import adr_pb2, adr_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SNETIntegration:
    """
    Integration layer between sleep prediction model and SingularityNET Risk Assessment service.
    
    This class handles:
    1. Integration with Photrek's Risk-Aware Assessment service
    2. Data format conversion for service compatibility
    3. Basic and enhanced balance management
    4. Service status monitoring
    5. Storage integration for evaluation results
    """
    
    def __init__(self, private_key: str, eth_rpc_endpoint: str):
        """
        Initialize the SingularityNET integration.
        
        Args:
            private_key: Ethereum private key for transactions
            eth_rpc_endpoint: Ethereum RPC endpoint URL
        """
        logger.info("Initializing SNETIntegration...")
        self.config = Config(
            private_key=private_key,
            eth_rpc_endpoint=eth_rpc_endpoint,
            concurrency=False  # Ensure sequential transaction handling
        )
        
        # Initialize SDK and service details
        self.sdk = SnetSDK(self.config)
        self.org_id = "Photrek"
        self.service_id = "risk-aware-assessment"
        self.group_name = "default_group"
        
        # Initialize components
        self.service_client = None
        self.service_details = None
        
        # Initialize service
        self._init_service()
    
    def _init_service(self) -> None:
        """Initialize service client and get payment details."""
        try:
            logger.info(f"Creating service client for {self.org_id}/{self.service_id}")
            self.service_client = self.sdk.create_service_client(
                org_id=self.org_id,
                service_id=self.service_id,
                group_name=self.group_name
            )
            
            # Get service payment details
            self.service_details = self.service_client.group
            
            # Log service configuration
            logger.info("Service client created successfully")
            logger.debug(f"Service pricing: {self.service_details['pricing']}")
            logger.debug(f"Payment group: {self.service_details['group_id']}")
            
        except Exception as e:
            logger.error(f"Service initialization error: {str(e)}", exc_info=True)
            raise ConnectionError(f"Failed to initialize Risk Assessment service: {str(e)}")

    def evaluate_predictions(self, 
                           predictions: List[List[float]], 
                           actuals: List[int]) -> Dict[str, float]:
        """
        Evaluate predictions using the Risk Assessment service.
        Maintains backward compatibility with V1 while using new format handling.
        
        Args:
            predictions: List of prediction probability vectors
            actuals: List of actual outcomes (as indices)
            
        Returns:
            Dictionary containing assessment metrics
        """
        try:
            # Validate inputs (maintaining V1 validation)
            if len(predictions) != len(actuals):
                raise ValueError("Number of predictions must match number of actuals")
                
            # Convert V1 format to storage format
            predictions_batch = {
                'batch_id': f'direct_eval_{datetime.now().isoformat()}',
                'predictions': [
                    {
                        'probabilities': {
                            'Poor': float(probs[0]),
                            'Moderate': float(probs[1]),
                            'Good': float(probs[2])
                        },
                        'actual_category': self._index_to_category(actual)
                    }
                    for probs, actual in zip(predictions, actuals)
                ],
                'prediction_ids': [f'direct_{i}' for i in range(len(predictions))]
            }
            
            # Use new format handling and service call
            formatted_input = self._convert_to_photrek_format(predictions_batch)
            return self._call_photrek_service(formatted_input)
                
        except Exception as e:
            logger.error(f"Service call error: {str(e)}")
            raise RuntimeError(f"Risk Assessment service call failed: {str(e)}")

    def evaluate_predictions_from_csv(self, csv_path: str) -> Dict[str, float]:
        """
        Evaluate predictions from CSV file.
        
        Args:
            csv_path: Path to CSV file containing predictions and actuals
            
        Returns:
            Dictionary with ADR metrics
        """
        try:
            logger.info(f"Reading CSV file: {csv_path}")
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Extract predictions into list format
            predictions = []
            for _, row in df.iterrows():
                pred_probs = [
                    float(row['prediction_poor']),
                    float(row['prediction_moderate']),
                    float(row['prediction_good'])
                ]
                predictions.append(pred_probs)
            
            # Extract actuals
            actuals = df['actual'].tolist()
            
            logger.debug(f"Extracted {len(predictions)} predictions from CSV")
            return self.evaluate_predictions(predictions, actuals)
            
        except Exception as e:
            logger.error(f"CSV processing error: {str(e)}")
            raise RuntimeError(f"Failed to process CSV file: {str(e)}")
        
    def process_predictions_for_evaluation(self, predictions_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process predictions from storage for evaluation by Photrek service.
        Primary method for storage-based evaluation flow.
        
        Flow:
        1. Get predictions batch from storage
        2. Format for Photrek service
        3. Get ADR metrics
        4. Format results for storage/calibration
        
        Args:
            predictions_batch: Dictionary from storage_v3.get_predictions_for_evaluation()
                Format: {
                    'batch_id': str,
                    'predictions': [{
                        'probabilities': {'Poor': float, 'Moderate': float, 'Good': float},
                        'actual_category': str
                    }],
                    'prediction_ids': List[str]
                }
        
        Returns:
            Dictionary for storage and calibration with ADR metrics
        """
        try:
            logger.info(f"Processing batch {predictions_batch['batch_id']} for evaluation")
            
            # Format and get metrics
            formatted_input = self._convert_to_photrek_format(predictions_batch)
            adr_metrics = self._call_photrek_service(formatted_input)
            
            # Format for storage
            return self._format_for_storage(
                metrics=adr_metrics,
                batch_id=predictions_batch['batch_id'],
                prediction_ids=predictions_batch['prediction_ids']
            )
            
        except Exception as e:
            logger.error(f"Error processing predictions: {str(e)}")
            raise RuntimeError(f"Failed to process predictions: {str(e)}")

    def _convert_to_photrek_format(self, predictions_batch: Dict[str, Any]) -> str:
        """
        Convert predictions to Photrek's expected format.
        
        Format required by Photrek: "numRows,numCols,base64EncodedCSV"
        Where CSV has no header and rows are: prob1,prob2,prob3,category(1-based)
        """
        try:
            # Convert category strings to 1-based indices
            category_map = {
                'Poor': 1,
                'Moderate': 2,
                'Good': 3
            }
            
            # Create CSV rows
            rows = []
            for pred in predictions_batch['predictions']:
                # Get probabilities in fixed order
                probs = [
                    str(pred['probabilities']['Poor']),
                    str(pred['probabilities']['Moderate']),
                    str(pred['probabilities']['Good'])
                ]
                
                # Add category (1-based index)
                category_idx = category_map[pred['actual_category']]
                row = ','.join(probs + [str(category_idx)])
                rows.append(row)
            
            # Create CSV content
            csv_content = '\n'.join(rows)
            
            # Get dimensions
            num_rows = len(rows)
            num_cols = 4  # 3 probs + 1 category
            
            # Encode to base64
            csv_bytes = csv_content.encode('utf-8')
            base64_content = base64.b64encode(csv_bytes).decode('utf-8')
            
            # Format final string
            return f"{num_rows},{num_cols},{base64_content}"
            
        except Exception as e:
            logger.error(f"Error converting to Photrek format: {str(e)}")
            raise ValueError(f"Failed to convert predictions format: {str(e)}")

    def _call_photrek_service(self, formatted_input: str) -> Dict[str, float]:
        """
        Call Photrek service with formatted input data.
        
        Args:
            formatted_input: String in format "numRows,numCols,base64EncodedCSV"
            
        Returns:
            Dictionary containing ADR metrics and additional service data
            
        Raises:
            ValueError: If input format or response is invalid
            RuntimeError: If service call fails
        """
        try:
            # Validate input before service call
            validated_input = self._validate_photrek_input(formatted_input)
            
            # Create service input
            input_str = adr_pb2.InputString(s=validated_input)
            
            # Get service stub
            stub = adr_pb2_grpc.ServiceDefinitionStub(self.service_client.grpc_channel)
            
            # Make service call with timeout
            response = stub.adr(input_str, timeout=30)  # 30 second timeout
            logger.info("Received response from Photrek service")
            
            # Verify basic response fields exist
            if not all(hasattr(response, field) for field in ['a', 'd', 'r']):
                raise ValueError("Invalid response format from service")
            
            # Build complete response dictionary
            result = {
                'accuracy': float(response.a),
                'decisiveness': float(response.d),
                'robustness': float(response.r),
                'num_rows': int(response.numr),
                'num_cols': int(response.numc)
            }
            
            # Add visualization if provided
            if hasattr(response, 'img') and response.img:
                result['visualization'] = response.img
                
            logger.debug(f"Processed service response: {result}")
            return result
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            logger.error(f"Error calling Photrek service: {str(e)}")
            raise RuntimeError(f"Photrek service call failed: {str(e)}")
    
    def _validate_photrek_input(self, formatted_input: str) -> str:
        """
        Validate and normalize input format before sending to service.
        Ensures exact compatibility with Photrek's server processing.
        
        Args:
            formatted_input: String in format "numRows,numCols,base64EncodedCSV"
            
        Returns:
            Validated and potentially normalized input string
            
        Raises:
            ValueError: If input format is invalid
        """
        try:
            # Split input into components
            parts = formatted_input.split(',', 2)
            if len(parts) != 3:
                raise ValueError("Input must have exactly 3 parts: numRows,numCols,base64data")
            
            # Validate numeric parts
            num_rows = int(parts[0])
            num_cols = int(parts[1])
            
            if num_rows <= 0 or num_cols <= 0:
                raise ValueError("Row and column counts must be positive")
                
            if num_cols != 4:  # Must be 3 probabilities + 1 category
                raise ValueError("Number of columns must be 4")
            
            # Process base64 content
            try:
                csv_content = base64.b64decode(parts[2]).decode('utf-8')
                rows = csv_content.strip().split('\n')
                
                if len(rows) != num_rows:
                    raise ValueError(f"Declared row count ({num_rows}) doesn't match actual rows ({len(rows)})")
                
                # Process and validate each row
                normalized_rows = []
                for i, row in enumerate(rows):
                    values = row.split(',')
                    if len(values) != 4:
                        raise ValueError(f"Row {i+1} has {len(values)} values, expected 4")
                    
                    # Convert probabilities to float64 and validate
                    try:
                        probs = [np.float64(v) for v in values[:3]]
                        
                        # Check probability range
                        if not all(0 <= p <= 1 for p in probs):
                            raise ValueError(f"Row {i+1} contains probabilities outside [0,1] range")
                        
                        # Handle zero probabilities as server does
                        if 0 in probs:
                            nonzero_probs = [p for p in probs if p > 0]
                            if nonzero_probs:
                                second_smallest = sorted(set(nonzero_probs))[0]
                                floor_value = second_smallest ** 2
                                probs = [floor_value if p == 0 else p for p in probs]
                        
                        # Normalize probabilities
                        total = sum(probs)
                        if total != 1.0:
                            probs = [p/total for p in probs]
                        
                        # Validate normalization
                        if not np.isclose(sum(probs), 1.0, rtol=1e-5):
                            raise ValueError(f"Failed to normalize probabilities in row {i+1}")
                        
                    except ValueError as e:
                        raise ValueError(f"Invalid probability format in row {i+1}: {str(e)}")
                    
                    # Validate category (must be 1-based index)
                    try:
                        category = int(values[3])
                        if category not in [1, 2, 3]:  # Valid categories are 1, 2, 3
                            raise ValueError(f"Invalid category in row {i+1}: {category}")
                    except ValueError:
                        raise ValueError(f"Invalid category format in row {i+1}")
                    
                    # Create normalized row
                    normalized_row = ','.join([f"{p:.10f}" for p in probs] + [values[3]])
                    normalized_rows.append(normalized_row)
                
                # Create new normalized CSV content
                normalized_csv = '\n'.join(normalized_rows)
                normalized_base64 = base64.b64encode(normalized_csv.encode('utf-8')).decode('utf-8')
                
                # Return normalized input
                return f"{num_rows},{num_cols},{normalized_base64}"
                        
            except (base64.binascii.Error, UnicodeDecodeError) as e:
                raise ValueError(f"Invalid base64 encoding: {str(e)}")
                
        except ValueError as e:
            logger.error(f"Input validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected validation error: {str(e)}")
            raise ValueError(f"Input validation failed: {str(e)}")

    def _format_for_storage(self, 
                          metrics: Dict[str, float],
                          batch_id: str,
                          prediction_ids: List[str]) -> Dict[str, Any]:
        """
        Format ADR metrics for storage and calibration use.
        """
        return {
            'batch_id': batch_id,
            'prediction_ids': prediction_ids,
            'accuracy': metrics['accuracy'],
            'decisiveness': metrics['decisiveness'],
            'robustness': metrics['robustness'],
            'num_predictions': len(prediction_ids),
            'timestamp': datetime.now().isoformat()
        }

    def _index_to_category(self, index: int) -> str:
        """Convert numeric index to category name."""
        categories = ['Poor', 'Moderate', 'Good']
        if 0 <= index < len(categories):
            return categories[index]
        raise ValueError(f"Invalid category index: {index}")

    def check_balance(self) -> float:
        """
        Check the AGIX token balance for service calls.
        Maintains V1's simple interface while using enhanced checking.
        """
        try:
            # Get all balances using enhanced checking
            balances = self._check_all_balances()
            # Return only escrow balance to maintain V1 compatibility
            return float(balances['mpe_balance'])
        except Exception as e:
            raise RuntimeError(f"Failed to check balance: {str(e)}")

    def _check_all_balances(self) -> Dict[str, float]:
        """Enhanced balance checking with all relevant balances."""
        try:
            account_address = self.sdk.account.address
            mpe_address = self.sdk.mpe_contract.contract.address
            token_contract = self.sdk.account.token_contract
            
            return {
                'eth_balance': float(self.sdk.web3.from_wei(
                    self.sdk.web3.eth.get_balance(account_address), 'ether')),
                'agix_balance': float(token_contract.functions.balanceOf(account_address).call()),
                'mpe_balance': float(self.sdk.account.escrow_balance()),
                'mpe_allowance': float(token_contract.functions.allowance(
                    account_address,
                    mpe_address
                ).call()),
                'account_address': account_address,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error checking balances: {str(e)}")
            raise RuntimeError(f"Failed to check balances: {str(e)}")

    def deposit_tokens(self, amount: float) -> bool:
        """Deposit AGIX tokens for service usage."""
        try:
            account = self.sdk.account
            account.deposit_to_escrow_account(amount)
            return True
        except Exception as e:
            raise RuntimeError(f"Token deposit failed: {str(e)}")
            
    def get_service_status(self) -> Dict[str, Any]:
        """Check Risk Assessment service status."""
        try:
            metadata = self.sdk.get_service_metadata(
                org_id=self.org_id,
                service_id=self.service_id
            )
            
            # Get current block for reference
            current_block = self.sdk.web3.eth.block_number
            
            status = {
                'status': 'available',
                'version': metadata.get('version', 'unknown'),
                'endpoints': metadata.get_all_endpoints_for_group(self.group_name),
                'current_block': current_block,
                'service_details': self.service_details
            }
            
            return status
            
        except Exception as e:
            return {
                'status': 'unavailable',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }