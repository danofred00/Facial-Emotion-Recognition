
import numpy as np

from fastapi import UploadFile
from os import environ as env
from os.path import join


def move_uploaded_file(file: UploadFile, path: str) -> str:
    """
        This move an uploaded file into a path, and it return the
        full path of the file saved 
    """
    
    fullpath = join(get_upload_path(), path)

    with open(fullpath, 'wb') as fp:
        fp.write(file.file.read())
        fp.close()

    return fullpath


def get_upload_path() -> str:
    return env.get('UPLOAD_PATH', 'storage/uploads')

def tolist(data) -> list:
    result = []
    
    for element in data:
        result.append(np.float32(element).item())

    print('result', result)
    return result