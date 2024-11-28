
from typing import List
from pydantic import BaseModel

class EmotionPrediction(BaseModel):
    emotion_id: int
    emotion_name: str

class PredictionResult(BaseModel):
    data: List[float] = []
    prediction: EmotionPrediction
