"""API routers for oratory video/audio analysis endpoints."""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, File, UploadFile, WebSocket, HTTPException, status

# Import the dependency, service, and exceptions
from api.services.video_processor import VideoProcessor
from api.services.exceptions import VideoProcessingError
from .video_processing import get_video_processor  # Re-use the dependency function

# Import the websocket handler
from api.websockets.video import handle_realtime_feedback

# Configure module logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Oratoria"])


@router.post("/feedback-oratoria", response_model=Dict[str, Any])
async def feedback_oratoria(
        video_file: UploadFile = File(...),
        quality_mode: str = "balanced",
        # CORRECT: Use the dependency directly in this endpoint
        processor: VideoProcessor = Depends(get_video_processor),
):
    """
    Receives a video, processes it, and returns comprehensive feedback.

    This endpoint analyzes both the visual and audio components of the video
    using the shared VideoProcessor service.

    Args:
        video_file: The video file to be analyzed.
        quality_mode: The analysis profile ('speed', 'balanced', 'quality').
        processor: Injected VideoProcessor dependency for handling the logic.

    Returns:
        A dictionary containing the oratory feedback and analysis results.
    """
    # The logic is now self-contained within this endpoint, using the processor
    try:
        if not video_file.filename or not video_file.content_type.startswith("video/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Invalid file format. Please upload a video file.",
            )

        # Call the service method directly on the injected processor
        results = await processor.process(video_file, quality_mode)
        return results

    except VideoProcessingError as e:
        logger.error(f"A processing error occurred for video {e.video_id}: {e.message}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process video: {e.message}",
        )
    except Exception as e:
        video_id = processor.video_id if isinstance(processor, VideoProcessor) else "unknown"
        logger.error(f"An unexpected error occurred for video {video_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred.",
        )


@router.websocket("/feedback-realtime")
async def feedback_realtime(
        websocket: WebSocket,
        buffer_size: int = 10,
        width: int = 160,
        height: int = 160,
        min_detection_confidence: float = 0.6,
):
    """
    WebSocket endpoint for real-time video analysis during recording.
    """
    await handle_realtime_feedback(
        websocket,
        buffer_size=buffer_size,
        width=width,
        height=height,
        min_detection_confidence=min_detection_confidence,
    )


@router.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "API de Feedback de Oratoria activa"}
