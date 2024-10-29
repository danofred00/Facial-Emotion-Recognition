
from fastapi.staticfiles import StaticFiles
from pathlib import Path

def get_static_files(prefix :str = '.'):
    return StaticFiles(directory=Path(f'{prefix}/'), html=True)