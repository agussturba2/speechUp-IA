
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import router

# Create FastAPI app instance with metadata
app = FastAPI(
    title="API de Feedback de Oratoria",
    description="API para análisis de video y generación de feedback de oratoria",
    version="1.0.0"
)

# Add CORS middleware for web client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main router
app.include_router(router)

