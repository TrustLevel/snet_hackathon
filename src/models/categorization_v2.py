#src/models/categorization_v2.py

from typing import Union, Tuple
import numpy as np

def validate_numeric_input(value: Union[int, float], name: str, min_val: float, max_val: float) -> None:
    """
    Validate numeric input is within acceptable range.
    
    Args:
        value: Numeric value to validate
        name: Name of the field for error messages
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
    Raises:
        TypeError: If value is not numeric
        ValueError: If value is outside acceptable range
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}")

def categorize_gender(gender: str) -> str:
    """
    Categorize gender into Male or Female.
    
    Args:
        gender: Input gender string
    Returns:
        str: 'Male' or 'Female'
    Raises:
        TypeError: If gender is not a string
        ValueError: If gender cannot be categorized
    """
    if not isinstance(gender, str):
        raise TypeError("Gender must be a string")
    
    gender = gender.lower().strip()
    if gender in ['male', 'm']:
        return "Male"
    elif gender in ['female', 'f']:
        return "Female"
    else:
        raise ValueError("Invalid gender value. Use 'Male'/'Female' or 'M'/'F'")

def categorize_age(age: Union[int, float]) -> str:
    """
    Categorize age into Young Adults, Middle-Aged, or Older Adults.
    
    Args:
        age: Age in years (18-120)
    Returns:
        str: Age category
    """
    validate_numeric_input(age, "Age", 18, 120)
    
    if 18 <= age <= 25:
        return "Young Adults"
    elif 26 <= age <= 64:
        return "Middle-Aged"
    else:  # 65-120
        return "Older Adults"

def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    Calculate Body Mass Index (BMI).
    
    Args:
        weight_kg: Weight in kilograms (30-300)
        height_m: Height in meters (1.0-2.5)
    Returns:
        float: BMI value
    """
    validate_numeric_input(weight_kg, "Weight", 30, 300)
    validate_numeric_input(height_m, "Height", 1.0, 2.5)
    
    return weight_kg / (height_m ** 2)

def categorize_bmi(bmi: float) -> str:
    """
    Categorize BMI into weight categories.
    
    Args:
        bmi: Body Mass Index value (10-60)
    Returns:
        str: BMI category
    """
    validate_numeric_input(bmi, "BMI", 10, 60)
    
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def categorize_rhr(rhr: Union[int, float]) -> str:
    """
    Categorize Resting Heart Rate (RHR) into Low, Moderate, High.
    
    Args:
        rhr: Resting heart rate in beats per minute (30-150)
    Returns:
        str: RHR category
    """
    validate_numeric_input(rhr, "Resting Heart Rate", 30, 150)
    
    if rhr < 60:
        return "Low"
    elif 60 <= rhr <= 80:
        return "Moderate"
    else:
        return "High"

def categorize_sleep_duration(duration: float) -> str:
    """
    Categorize average sleep duration into Short, Moderate, Long.
    
    Args:
        duration: Sleep duration in hours (0-24)
    Returns:
        str: Sleep duration category
    """
    validate_numeric_input(duration, "Sleep Duration", 0, 24)
    
    if duration < 6.0:
        return "Short"
    elif 6.0 <= duration <= 8.0:
        return "Moderate"
    else:
        return "Long"

def categorize_stress_level(stress: Union[str, int, float]) -> str:
    """
    Categorize stress level into Low, Moderate, High.
    
    Args:
        stress: Stress level (0-10) or category string
    Returns:
        str: Stress category
    """
    if isinstance(stress, str):
        stress = stress.strip()
        if stress in ["Low", "Moderate", "High"]:
            return stress
        try:
            stress = float(stress)
        except ValueError:
            raise ValueError("Invalid stress level string. Use 'Low', 'Moderate', 'High' or numeric value 0-10")
    
    validate_numeric_input(float(stress), "Stress Level", 0, 10)
    
    if float(stress) <= 4:
        return "Low"
    elif float(stress) <= 7:
        return "Moderate"
    else:
        return "High"

def categorize_blue_light_exposure(hours_without_bluelight: Union[int, float]) -> str:
    """
    Categorize blue light exposure based on hours before bed.
    
    Args:
        hours_without_bluelight: Hours without blue light exposure (0-3)
    Returns:
        str: Blue light exposure category
    """
    validate_numeric_input(hours_without_bluelight, "Hours without blue light", 0, 3)
    
    if hours_without_bluelight <= 0.5:
        return "High Exposure"
    elif hours_without_bluelight <= 1.5:
        return "Medium Exposure"
    elif hours_without_bluelight <= 2.5:
        return "Low Exposure"
    else:
        return "No Exposure"

def categorize_sleep_quality(quality: Union[int, float]) -> str:
    """
    Categorize sleep quality score into Poor, Moderate, Good.
    
    Args:
        quality: Sleep quality score (0-10)
    Returns:
        str: Sleep quality category
    """
    validate_numeric_input(quality, "Sleep Quality", 0, 10)
    
    if quality <= 5:
        return "Poor"
    elif quality <= 7:
        return "Moderate"
    else:
        return "Good"