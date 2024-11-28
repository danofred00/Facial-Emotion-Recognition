
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse

from packages.backend.model import load_model, image_to_x
from packages.backend.model.constants import EMOTION_MAP, MODEL_PATH
from .utils import move_uploaded_file, tolist
from .types import PredictionResult

import re

router = APIRouter()
model = load_model(MODEL_PATH)


@router.get("/")
def root_endpoint():
    return {
        "endpoints": {
            "/api/v1/predict_emotion": "Use to perform a prediction task"
        }
    }

@router.get("/emotion_map")
def emotion_map():
    return {
        "data": EMOTION_MAP
    }
    

@router.post('/predict_emotion', response_model=PredictionResult)
def predict_emotion(image: UploadFile):
    """
        This can predict the user emotion
    """

    if not re.match("^image/(.*)$", image.content_type):
        return JSONResponse(status_code=422, content={"message": "Invalid image type"})

    path = move_uploaded_file(image, f"{image.filename}")
    try :
        image_x = image_to_x(path)
        result = predict(image_x)

        return result
    except RuntimeError:
        return JSONResponse(
            status_code=422,
            content={"message": "No face detected inside the image"}
        )


def predict(x, **kwargs):
    result = model.predict(x, use_multiprocessing=True, **kwargs)
    result = result[0]
    emotion = result.argmax()
    
    return {
        "data": tolist(result),
        "prediction": {
            "emotion_id": emotion,
            "emotion_name": EMOTION_MAP[emotion]
        }
    }