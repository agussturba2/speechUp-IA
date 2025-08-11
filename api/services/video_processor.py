# api/services/video_processor.py
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from starlette.concurrency import run_in_threadpool

from api.services import constants as k
from api.services.exceptions import AudioAnalysisError
from api.utils import file_handling
from audio import OratoryAnalyzer
from video.pipeline import run_analysis_pipeline

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Encapsulates the logic for saving, analyzing, and generating feedback for a video.
    """

    def __init__(
            self,
            video_analyzer_func: Callable = run_analysis_pipeline,
            audio_analyzer_class: Type[OratoryAnalyzer] = OratoryAnalyzer,
    ):
        """
        Initializes the processor with injectable analysis components.

        Args:
            video_analyzer_func: The function to run for video analysis.
            audio_analyzer_class: The class to instantiate for audio analysis.
        """
        self.video_id = str(uuid.uuid4())
        self.temp_dir = Path(tempfile.gettempdir())
        self.temp_video_path = self.temp_dir / f"{self.video_id}.mp4"
        self.temp_audio_clone_path: Optional[Path] = None

        self._video_analyzer = video_analyzer_func
        self._audio_analyzer_cls = audio_analyzer_class
        self._ffmpeg_available, self._ffprobe_available = file_handling.check_ffmpeg_tools()
        logger.info(f"Initialized VideoProcessor with ID: {self.video_id}")

    async def process(self, video_file: Any, quality_mode: str) -> Dict[str, Any]:
        """
        Orchestrates the full video processing pipeline.

        Args:
            video_file: The uploaded video file object.
            quality_mode: The quality profile for video analysis.

        Returns:
            A dictionary containing the combined analysis results.
        """
        await file_handling.save_temp_video(self.temp_video_path, video_file)
        file_handling.validate_video_file(self.temp_video_path)

        video_results = await self._run_video_analysis(quality_mode)
        audio_results = await self._run_audio_analysis()

        combined_results = {
            k.VIDEO_ANALYSIS_KEY: video_results,
            **audio_results,  # Merge audio results (analysis, warning, or error)
        }
        return combined_results

    async def _run_video_analysis(self, quality_mode: str) -> Dict[str, Any]:
        """Runs the video analysis pipeline in a thread pool."""
        logger.info(f"[{self.video_id}] Starting video analysis...")
        results = await run_in_threadpool(
            self._video_analyzer,
            video_path=str(self.temp_video_path),
            quality_mode=quality_mode,
        )
        logger.info(f"[{self.video_id}] Video analysis completed.")
        return results

    async def _run_audio_analysis(self) -> Dict[str, Any]:
        """Runs the audio analysis pipeline in a thread pool if tools are available."""
        if not self._ffmpeg_available or not self._ffprobe_available:
            return {k.AUDIO_WARNING_KEY: "Audio analysis skipped: FFmpeg/FFprobe not found."}

        self._clone_video_for_audio()
        if not self.temp_audio_clone_path:
            return {k.AUDIO_ERROR_KEY: "Audio analysis skipped due to file copy failure."}

        logger.info(f"[{self.video_id}] Starting audio analysis...")
        try:
            analyzer = self._audio_analyzer_cls()
            results = await run_in_threadpool(
                analyzer.analyze, media_path=str(self.temp_audio_clone_path)
            )
            logger.info(f"[{self.video_id}] Audio analysis completed.")
            return {k.AUDIO_ANALYSIS_KEY: results}
        except Exception as e:
            logger.error(f"[{self.video_id}] Audio analysis failed: {e}", exc_info=True)
            raise AudioAnalysisError(f"Audio analysis failed: {e}", self.video_id)

    def _clone_video_for_audio(self) -> None:
        """Creates a copy of the video file for the audio analyzer to use safely."""
        self.temp_audio_clone_path = self.temp_dir / f"{self.video_id}_audio.mp4"
        try:
            shutil.copy2(self.temp_video_path, self.temp_audio_clone_path)
            logger.info(f"[{self.video_id}] Cloned video for audio analysis.")
        except Exception as e:
            logger.warning(f"[{self.video_id}] Failed to clone video for audio: {e}")
            self.temp_audio_clone_path = None

    def cleanup(self) -> None:
        """Removes all temporary files created during processing."""
        logger.info(f"[{self.video_id}] Cleaning up temporary files.")
        file_handling.cleanup_temp_files(self.temp_video_path, self.temp_audio_clone_path)
