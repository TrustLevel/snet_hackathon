# src/models/cpts_v2.py

import numpy as np
from typing import Dict

def validate_cpt_probabilities(cpt: Dict[str, Dict[str, float]], name: str) -> None:
    """
    Validate that probabilities in CPT sum to 1.0 for each condition.
    
    Args:
        cpt: Conditional Probability Table
        name: Name of the CPT for error messages
    Raises:
        ValueError: If probabilities don't sum to 1.0
    """
    for condition, probs in cpt.items():
        total = sum(probs.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise ValueError(
                f"CPT '{name}': Probabilities for condition '{condition}' "
                f"sum to {total}, not 1.0"
            )

def gender_sleep_quality_cpt() -> Dict[str, Dict[str, float]]:
    """
    CPT for Sleep Quality influenced by Gender.
    Research indicates differences in sleep patterns and quality between genders.
    
    Returns:
        Dict mapping gender categories to sleep quality probabilities
    """
    cpt = {
        "Male": {
            "Poor": 0.3,
            "Moderate": 0.4,
            "Good": 0.3
        },
        "Female": {
            "Poor": 0.2,
            "Moderate": 0.5,
            "Good": 0.3
        }
    }
    validate_cpt_probabilities(cpt, "gender_sleep_quality")
    return cpt

def age_sleep_quality_cpt() -> Dict[str, Dict[str, float]]:
    """
    CPT for Sleep Quality influenced by Age.
    Sleep patterns and quality change significantly across age groups.
    
    Returns:
        Dict mapping age categories to sleep quality probabilities
    """
    cpt = {
        "Young Adults": {
            "Poor": 0.1,
            "Moderate": 0.2,
            "Good": 0.7
        },
        "Middle-Aged": {
            "Poor": 0.3,
            "Moderate": 0.4,
            "Good": 0.3
        },
        "Older Adults": {
            "Poor": 0.4,
            "Moderate": 0.4,
            "Good": 0.2
        }
    }
    validate_cpt_probabilities(cpt, "age_sleep_quality")
    return cpt

def bmi_sleep_quality_cpt() -> Dict[str, Dict[str, float]]:
    """
    CPT for Sleep Quality influenced by BMI.
    BMI can significantly impact sleep quality through various physiological mechanisms.
    
    Returns:
        Dict mapping BMI categories to sleep quality probabilities
    """
    cpt = {
        "Underweight": {
            "Poor": 0.4,
            "Moderate": 0.4,
            "Good": 0.2
        },
        "Normal": {
            "Poor": 0.1,
            "Moderate": 0.2,
            "Good": 0.7
        },
        "Overweight": {
            "Poor": 0.3,
            "Moderate": 0.4,
            "Good": 0.3
        },
        "Obese": {
            "Poor": 0.7,
            "Moderate": 0.2,
            "Good": 0.1
        }
    }
    validate_cpt_probabilities(cpt, "bmi_sleep_quality")
    return cpt

def rhr_sleep_quality_cpt() -> Dict[str, Dict[str, float]]:
    """
    CPT for Sleep Quality influenced by Resting Heart Rate (RHR).
    RHR can indicate overall cardiovascular health and stress levels.
    
    Returns:
        Dict mapping RHR categories to sleep quality probabilities
    """
    cpt = {
        "Low": {
            "Poor": 0.1,
            "Moderate": 0.2,
            "Good": 0.7
        },
        "Moderate": {
            "Poor": 0.3,
            "Moderate": 0.4,
            "Good": 0.3
        },
        "High": {
            "Poor": 0.7,
            "Moderate": 0.2,
            "Good": 0.1
        }
    }
    validate_cpt_probabilities(cpt, "rhr_sleep_quality")
    return cpt

def sleep_duration_sleep_quality_cpt() -> Dict[str, Dict[str, float]]:
    """
    CPT for Sleep Quality influenced by Average Sleep Duration.
    Sleep duration is one of the most direct indicators of sleep quality.
    
    Returns:
        Dict mapping sleep duration categories to sleep quality probabilities
    """
    cpt = {
        "Short": {
            "Poor": 0.7,
            "Moderate": 0.2,
            "Good": 0.1
        },
        "Moderate": {
            "Poor": 0.2,
            "Moderate": 0.4,
            "Good": 0.4
        },
        "Long": {
            "Poor": 0.1,
            "Moderate": 0.3,
            "Good": 0.6
        }
    }
    validate_cpt_probabilities(cpt, "sleep_duration_sleep_quality")
    return cpt

def stress_sleep_quality_cpt() -> Dict[str, Dict[str, float]]:
    """
    CPT for Sleep Quality influenced by Stress Level.
    Stress has a significant impact on sleep through hormonal and psychological mechanisms.
    
    Returns:
        Dict mapping stress level categories to sleep quality probabilities
    """
    cpt = {
        "Low": {
            "Poor": 0.1,
            "Moderate": 0.1,
            "Good": 0.8
        },
        "Moderate": {
            "Poor": 0.2,
            "Moderate": 0.5,
            "Good": 0.3
        },
        "High": {
            "Poor": 0.7,
            "Moderate": 0.2,
            "Good": 0.1
        }
    }
    validate_cpt_probabilities(cpt, "stress_sleep_quality")
    return cpt

def blue_light_sleep_quality_cpt() -> Dict[str, Dict[str, float]]:
    """
    CPT for Sleep Quality influenced by Blue Light Exposure.
    Blue light exposure before bedtime can significantly disrupt melatonin production.
    
    Returns:
        Dict mapping blue light exposure categories to sleep quality probabilities
    """
    cpt = {
        "High Exposure": {
            "Poor": 0.7,
            "Moderate": 0.2,
            "Good": 0.1
        },
        "Medium Exposure": {
            "Poor": 0.5,
            "Moderate": 0.3,
            "Good": 0.2
        },
        "Low Exposure": {
            "Poor": 0.1,
            "Moderate": 0.4,
            "Good": 0.5
        },
        "No Exposure": {
            "Poor": 0.1,
            "Moderate": 0.1,
            "Good": 0.8
        }
    }
    validate_cpt_probabilities(cpt, "blue_light_sleep_quality")
    return cpt