from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.websockets import handle_oratory_feedback, handle_incremental_oratory_feedback

# Cargar variables de entorno desde el archivo .env
load_dotenv()

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

# Register WebSocket endpoints
app.websocket("/ws/v1/feedback-oratoria")(handle_oratory_feedback)

app.websocket("/ws/v1/incremental-feedback")(handle_incremental_oratory_feedback)
