#src/utils/preprocessing.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.models.categorization_v2 import (
    categorize_gender,
    categorize_age,
    categorize_bmi,
    categorize_rhr,
    categorize_sleep_duration,
    categorize_stress_level,
    categorize_blue_light_exposure,
    categorize_sleep_quality,
    calculate_bmi
)

def validate_input_data(data: Dict[str, Any]) -> None:
    """
    Validate that all required fields are present and within acceptable ranges.
    
    Args:
        data: Dictionary containing input data
    Raises:
        ValueError: If required fields are missing or invalid
        TypeError: If fields have wrong type
    """
    # Required fields with their types and ranges
    required_fields = {
        "gender": (str, None),  # Will be validated in categorize_gender
        "age": ((int, float), (18, 120)),
        "weight": (float, (30, 300)),  # kg
        "height": (float, (1.0, 2.5)),  # meters
        "resting_heart_rate": ((int, float), (30, 150)),
        "sleep_duration": (float, (0, 24)),
        "stress_level": ((str, int, float), (0, 10)),
        "blue_light_hours": ((int, float), (0, 3))
    }
    
    for field, (expected_type, value_range) in required_fields.items():
        # Check presence
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
            
        # Check type
        if not isinstance(data[field], expected_type):
            raise TypeError(f"Invalid type for {field}. Expected {expected_type}")
            
        # Check range if applicable
        if value_range is not None and not isinstance(data[field], str):
            value = float(data[field])
            min_val, max_val = value_range
            if value < min_val or value > max_val:
                raise ValueError(f"{field} must be between {min_val} and {max_val}")


def preprocess_user_data(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Preprocess raw user input data into categorized format.
    
    Args:
        data: Dictionary containing raw user data
    Returns:
        Dictionary containing categorized data
    """
    # Validate all inputs first
    validate_input_data(data)
    
    # Handle height conversion if needed (already validated to be between 1.0 and 2.5m)
    height_m = data["height"]
    
    # Calculate BMI
    bmi = calculate_bmi(data["weight"], height_m)
    
    return {
        "gender": categorize_gender(data["gender"]),
        "age": categorize_age(data["age"]),
        "bmi": categorize_bmi(bmi),
        "rhr": categorize_rhr(data["resting_heart_rate"]),
        "sleep_duration": categorize_sleep_duration(data["sleep_duration"]),
        "stress_level": categorize_stress_level(data["stress_level"]),
        "blue_light": categorize_blue_light_exposure(data["blue_light_hours"])
    }

def preprocess_dataset(df: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Preprocess a dataset for model training or evaluation.
    
    Args:
        df: DataFrame containing sleep health dataset
    Returns:
        Tuple containing list of preprocessed instances and their sleep quality categories
    """
    validate_dataset(df)
    
    # Process each row into model format
    processed_data = []
    sleep_quality_labels = []
    
    for _, row in df.iterrows():
        # Convert row to standard format
        instance = {
            "gender": row["Gender"],
            "age": row["Age"],
            "weight": extract_weight(row),  # Need to implement this based on dataset
            "height": extract_height(row),  # Need to implement this based on dataset
            "resting_heart_rate": row["Heart Rate"],
            "sleep_duration": row["Sleep Duration"],
            "stress_level": row["Stress Level"],
            "blue_light_hours": estimate_blue_light_hours(row)  # Estimate from available data
        }
        
        try:
            processed = preprocess_user_data(instance)
            processed_data.append(processed)
            sleep_quality_labels.append(
                categorize_sleep_quality(row["Quality of Sleep"])
            )
        except (ValueError, TypeError) as e:
            print(f"Warning: Skipping row due to error: {e}")
            continue
    
    return processed_data, sleep_quality_labels

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate that the dataset contains all required columns.
    
    Args:
        df: DataFrame to validate
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = {
        "Gender",
        "Age",
        "Heart Rate",
        "Sleep Duration",
        "Quality of Sleep",
        "Stress Level"
    }
    
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def estimate_blue_light_hours(row: pd.Series) -> float:
    """
    Estimate blue light exposure hours before bed based on available data.
    This is a placeholder function - modify based on actual dataset features.
    
    Args:
        row: Series containing row data
    Returns:
        Estimated hours without blue light before bed
    """
    # This is a simplified estimation - modify based on actual data
    occupation = str(row.get("Occupation", "")).lower()
    physical_activity = float(row.get("Physical Activity Level", 50))
    
    if "engineer" in occupation or "software" in occupation:
        return 1.0  # High screen time
    elif physical_activity > 60:
        return 2.0  # More active, less screen time
    else:
        return 1.5  # Moderate screen time

def extract_weight(row: pd.Series) -> float:
    """
    Extract weight from BMI and height data if available.
    Placeholder function - modify based on actual dataset structure.
    
    Args:
        row: Series containing row data
    Returns:
        Weight in kg
    """
    # This is a placeholder - implement based on actual data
    return 70.0  # Default value, replace with actual logic

def extract_height(row: pd.Series) -> float:
    """
    Extract height from BMI and weight data if available.
    Placeholder function - modify based on actual dataset structure.
    
    Args:
        row: Series containing row data
    Returns:
        Height in meters
    """
    # This is a placeholder - implement based on actual data
    return 1.7  # Default value, replace with actual logic

def prepare_training_data(file_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Load and preprocess dataset from file for model training.
    
    Args:
        file_path: Path to the CSV file
    Returns:
        Tuple containing preprocessed data and labels
    """
    try:
        df = pd.read_csv(file_path)
        return preprocess_dataset(df)
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")