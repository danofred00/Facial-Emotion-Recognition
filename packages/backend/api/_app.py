
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def root_endpoint():
    return {"message": "Hello from Server"}