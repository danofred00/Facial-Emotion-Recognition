"""This package contains the API for the backend."""
from fastapi import APIRouter
from . import _app

api_router = APIRouter()

api_router.include_router(router=_app.router)
