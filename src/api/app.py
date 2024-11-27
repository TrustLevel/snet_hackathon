# src/api/app_v2.py - version 3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional
from datetime import datetime

from src.utils.preprocessing import preprocess_user_data
from src.utils.storage_v3 import SleepDataStorage
from src.models.sleep_quality_predict_v5 import SleepQualityPredict

app = FastAPI(
    title="Sleep Quality Prediction API",
    description="API for predicting sleep quality and recording feedback",
    version="2.0.0"
)

# Initialize components
storage = SleepDataStorage()
model = SleepQualityPredict()

class SleepQualityInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Male",
                "age": 30.0,
                "weight": 70.0,
                "height": 1.75,
                "resting_heart_rate": 65.0,
                "sleep_duration": 7.5,
                "stress_level": 5.0,
                "blue_light_hours": 1.0
            }
        }
    )

    gender: str = Field(..., description="Male or Female")
    age: float = Field(..., ge=18, le=120, description="Age in years")
    weight: float = Field(..., ge=30, le=300, description="Weight in kg")
    height: float = Field(..., ge=1.0, le=2.5, description="Height in meters")
    resting_heart_rate: float = Field(..., ge=30, le=150, description="Resting heart rate (bpm)")
    sleep_duration: float = Field(..., ge=0, le=24, description="Sleep duration in hours")
    stress_level: float = Field(..., ge=0, le=10, description="Stress level (0-10)")
    blue_light_hours: float = Field(..., ge=0, le=3, description="Hours without blue light before bed")

class SleepQualityOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    prediction_id: str
    probabilities: Dict[str, float]
    predicted_category: str
    timestamp: str

class SleepQualityFeedback(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    prediction_id: str
    actual_quality: float = Field(..., ge=0, le=10, description="Actual sleep quality (0-10)")

@app.post("/predict", response_model=SleepQualityOutput)
async def predict_sleep_quality(data: SleepQualityInput):
    """Make a sleep quality prediction."""
    try:
        input_data = data.model_dump()
        
        # Ensure numeric types
        input_data['age'] = float(input_data['age'])
        input_data['weight'] = float(input_data['weight'])
        input_data['height'] = float(input_data['height'])
        input_data['resting_heart_rate'] = float(input_data['resting_heart_rate'])
        input_data['sleep_duration'] = float(input_data['sleep_duration'])
        input_data['stress_level'] = float(input_data['stress_level'])
        input_data['blue_light_hours'] = float(input_data['blue_light_hours'])
        
        prediction = model.predict(input_data)
        
        return {
            "prediction_id": prediction["prediction_id"],
            "probabilities": prediction["probabilities"],
            "predicted_category": prediction["predicted_category"],
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def record_sleep_quality(feedback: SleepQualityFeedback):
    """Record actual sleep quality feedback."""
    try:
        result = model.record_actual_quality(
            prediction_id=feedback.prediction_id,
            actual_quality=float(feedback.actual_quality)
        )
        
        if not result:
            raise HTTPException(
                status_code=404, 
                detail=f"Prediction {feedback.prediction_id} not found"
            )
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "prediction_id": feedback.prediction_id,
            "actual_quality": feedback.actual_quality,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str):
    """Get details of a specific prediction."""
    try:
        predictions = storage.get_predictions_with_outcomes(limit=None)
        for pred in predictions:
            if pred['prediction_id'] == prediction_id:
                return pred
        raise HTTPException(
            status_code=404,
            detail=f"Prediction {prediction_id} not found"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check API health."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "storage": "healthy" if storage else "unavailable",
            "model": "healthy" if model else "unavailable"
        }
    }