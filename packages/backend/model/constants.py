"""
    This file is a part of the Facial Emotion Recognition project
    @author Danofred00
    @author Jonas
"""

# This represent the list of avaliable emotions
EMOTION_MAP = {
    0: 'Angry', 
    1: 'Digust', 
    2: 'Fear', 
    3: 'Happy', 
    4: 'Sad', 
    5: 'Surprise', 
    6: 'Neutral'
}

# this represent the default reshaped size of an image
DEFAULT_RESHAPE = (48, 48)

# the default path where the pretrained model is store
MODEL_PATH='packages/backend/model/data/facial_emotion_recognition_model.h5'