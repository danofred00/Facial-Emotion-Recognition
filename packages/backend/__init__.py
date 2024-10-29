"""Main module for the backend package."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import api_router

def get_app():
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import the API routes
    app.include_router(router=api_router, prefix="/api/v1")

    # Return the FastAPI instance
    return app