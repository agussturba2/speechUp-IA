# api/routers/video_processing.py
"""
API endpoint for processing video files and returning analysis.
"""
import logging
from typing import Any, Dict, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from api.services.exceptions import VideoProcessingError
from api.services.video_processor import VideoProcessor

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_video_processor() -> AsyncGenerator[VideoProcessor, None]:
    """
    FastAPI dependency to create and clean up a VideoProcessor instance.

    This pattern ensures that the processor's resources (like temporary files)
    are always released after the request is handled.

    Yields:
        The VideoProcessor instance to be used in the request.
    """
    processor = VideoProcessor()
    try:
        yield processor
    finally:
        # This cleanup code will run after the response has been sent
        processor.cleanup()


@router.post(
    "/process-video/",
    tags=["Video Processing"],
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid video file or processing failure"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected server error"},
    },
)
async def process_video(
        video_file: UploadFile,
        quality_mode: str = "balanced",
        processor: VideoProcessor = Depends(get_video_processor),
) -> Dict[str, Any]:
    """
    Processes an uploaded video file to generate oratory feedback.

    This endpoint handles:
    - Saving the video file temporarily.
    - Running parallel video and audio analysis pipelines.
    - Aggregating results and generating comprehensive feedback.

    Args:
        video_file: The video file to be analyzed (as `UploadFile`).
        quality_mode: The analysis profile ('speed', 'balanced', 'quality').
        processor: Injected VideoProcessor dependency for handling the logic.

    Returns:
        A dictionary with the combined analysis and feedback.

    Raises:
        HTTPException: If the file is invalid or a processing error occurs.
    """
    if not video_file.filename or not video_file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid file format. Please upload a video file.",
        )

    try:
        # The core logic is now cleaner and free of the try/finally block
        results = await processor.process(video_file, quality_mode)
        return results
    except VideoProcessingError as e:
        # This is safe because VideoProcessingError is guaranteed to have a video_id
        logger.error(f"A processing error occurred for video {e.video_id}: {e.message}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process video: {e.message}",
        )
    except Exception as e:
        # This block now safely handles errors that occur during dependency injection.
        video_id = "unknown"
        if isinstance(processor, VideoProcessor):
            # Only access video_id if the processor was successfully created.
            video_id = processor.video_id

        logger.error(f"An unexpected error occurred for video {video_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred.",
        )
