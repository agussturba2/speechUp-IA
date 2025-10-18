"""
WebSocket handlers for incremental oratory analysis and feedback.

This module provides a WebSocket handler that processes audio and video frames
incrementally as they are received, rather than waiting for a complete video file.

REFACTORED: Now uses modular components (SessionCoordinator, AudioProcessor, 
VideoBufferManager, MetricsAnalyzer) from Phase 2 refactoring.
"""

import logging
import os
import time
import json
from typing import Dict, Any, Optional
from pathlib import Path
import cv2
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

from video.pipeline import run_analysis_pipeline
from .session_coordinator import SessionCoordinator, SessionFactory
from .models import AnalysisResult, AnalysisStatus
from .oratory import process_pipeline_results, send_analysis_result
from .config import config as incremental_config

logger = logging.getLogger(__name__)


class IncrementalOratorySession:
    """
    Adapter for SessionCoordinator with backward compatibility.
    
    Wraps SessionCoordinator and adds final feedback generation logic.
    Maintains API compatibility with legacy code.
    """
    
    def __init__(
        self,
        buffer_size: int = 30,
        width: int = 640,
        height: int = 480,
        processing_interval: int = 60,
    ):
        """
        Initialize session adapter.
        
        Args:
            buffer_size: Size of the frame buffer
            width: Width of the video frames
            height: Height of the video frames
            processing_interval: Process frames every N frames
        """
        # Use SessionCoordinator internally
        self._coordinator = SessionCoordinator(
            width=width,
            height=height,
            enable_incremental=True
        )
        
        # Store config for compatibility
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.processing_interval = processing_interval
        
        logger.info(f"IncrementalOratorySession initialized (using SessionCoordinator)")
    
    # Delegate to SessionCoordinator
    def add_frame(self, frame_data: bytes) -> bool:
        """Add video frame."""
        return self._coordinator.add_frame(frame_data)
    
    def add_audio(self, audio_data: bytes) -> None:
        """Add audio data."""
        self._coordinator.add_audio(audio_data)
    
    async def process_incremental(self) -> Dict[str, Any]:
        """
        Process accumulated data incrementally.
        
        Returns:
            Dict with incremental update (converted to dict for compatibility)
        """
        update = await self._coordinator.process_incremental()
        
        # Convert Pydantic model to dict for backward compatibility
        result_dict = {
            "status": update.status.value if hasattr(update.status, 'value') else update.status,
            "frames_processed": update.frames_processed,
            "buffer_size": update.buffer_size,
            "processing_time_sec": update.processing_time_sec,
            "timestamp": update.timestamp,
        }
        
        # Add incremental metrics if present
        if update.incremental_metrics:
            result_dict["incremental_metrics"] = {
                "wpm": round(update.incremental_metrics.wpm, 1),
                "fillers_per_min": round(update.incremental_metrics.fillers_per_min, 2),
                "gesture_rate": round(update.incremental_metrics.gesture_rate, 2),
                "expression_variability": round(update.incremental_metrics.expression_variability, 2),
            }
            result_dict["session_duration"] = update.session_duration
            result_dict["confidence"] = update.confidence
        
        # Add recent events
        if update.recent_fillers:
            result_dict["recent_fillers"] = [f.model_dump() if hasattr(f, 'model_dump') else f 
                                             for f in update.recent_fillers]
        if update.recent_gestures:
            result_dict["recent_gestures"] = [g.model_dump() if hasattr(g, 'model_dump') else g 
                                              for g in update.recent_gestures]
        
        # Add error info if present
        if update.error:
            result_dict["error"] = update.error
            result_dict["error_type"] = update.error_type
        
        if update.message:
            result_dict["message"] = update.message
        
        return result_dict
    
    def end_stream(self) -> None:
        """Signal stream end and finalize video."""
        self._coordinator.end_stream()
    
    async def generate_final_feedback(self) -> Dict[str, Any]:
        """
        Generate final feedback after stream ends.
        
        Uses the full analysis pipeline with accumulated data.
        
        Returns:
            Dict with complete feedback data
        """
        if self._coordinator.streaming_active:
            logger.warning("generate_final_feedback called while stream is still active")
            return {"status": "streaming", "message": "Stream still active"}
        
        if self._coordinator.analysis_in_progress:
            logger.warning("Analysis already in progress")
            return {"status": "analyzing", "message": "Analysis already in progress"}
        
        # Get video path
        video_path = self._coordinator.get_video_path()
        
        if not video_path or not os.path.exists(str(video_path)):
            logger.warning("No video file available, creating from frames...")
            video_path = self._coordinator.end_stream()
            
            if not video_path or not os.path.exists(str(video_path)):
                return {
                    "status": "error",
                    "error": "Failed to create video file for analysis",
                    "timestamp": time.time()
                }
        
        logger.info(f"Starting final analysis with video: {video_path}")
        
        try:
            # Verify video file
            file_size = os.path.getsize(str(video_path))
            logger.info(f"Video file size: {file_size} bytes")
            
            if file_size < incremental_config.min_video_file_size:
                raise ValueError(f"Video file too small: {file_size} bytes")
            
            # Verify video can be opened
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            if frame_count == 0:
                raise ValueError("Video has 0 frames")
            
            logger.info(f"Video verified: {frame_count} frames, {fps:.1f} FPS, {width}x{height}")
            
            # Export accumulated audio buffer
            audio_wav_path = self._coordinator.audio_processor.export_buffer_to_wav()

            # Run analysis pipeline
            logger.info("Running analysis pipeline")
            t0 = time.perf_counter()
            
            proc = run_analysis_pipeline(str(video_path), audio_path=audio_wav_path)
            analysis_time = time.perf_counter() - t0
            
            logger.info(f"Analysis completed in {analysis_time:.2f}s")
            
            # Process results
            result = process_pipeline_results(proc, analysis_time)
            
            if result is None:
                raise ValueError("Pipeline returned None result")
            
            # Enhance with incremental data
            result = self._enhance_with_incremental_data(result)
            
            # Add debug info
            if "quality" not in result:
                result["quality"] = {}
            if "debug" not in result["quality"]:
                result["quality"]["debug"] = {}
            
            result["quality"]["debug"].update({
                "incremental_processing": True,
                "session_coordinator": True,
                "total_frames_received": self._coordinator.get_frame_count()
            })
            
            logger.info("Final feedback generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating final feedback: {e}", exc_info=True)
            
            # Try to create result from incremental data as fallback
            try:
                logger.info("Attempting fallback result from incremental data")
                return self._create_fallback_result()
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                
                return {
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": time.time()
                }
        finally:
            try:
                if 'audio_wav_path' in locals() and audio_wav_path and os.path.exists(audio_wav_path):
                    os.remove(audio_wav_path)
                    logger.debug(f"Removed temporary audio file: {audio_wav_path}")
            except Exception as cleanup_exc:
                logger.warning(f"Failed to remove temporary audio file: {cleanup_exc}")
    
    def _enhance_with_incremental_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance final result with incremental metrics."""
        metrics = self._coordinator.metrics_analyzer.get_metrics()
        if metrics is None:
            logger.warning("No incremental metrics available")
            return result

        logger.info(f"Enhancing with incremental data: wpm={metrics.wpm:.1f}, fillers={metrics.fillers_per_min:.2f}, "
                   f"gesture_rate={metrics.gesture_rate:.2f}, expression_var={metrics.expression_variability:.2f}")

        # Store in debug
        quality = result.setdefault("quality", {})
        debug = quality.get("debug")
        if not isinstance(debug, dict):
            debug = {}
            quality["debug"] = debug

        debug["incremental_metrics"] = {
            "wpm": round(metrics.wpm, 1),
            "fillers_per_min": round(metrics.fillers_per_min, 2),
            "gesture_rate": round(metrics.gesture_rate, 2),
            "expression_variability": round(metrics.expression_variability, 2)
        }

        # Override main metrics with incremental data if they are more reliable
        verbal = result.setdefault("verbal", {})
        
        # Use incremental WPM if pipeline returned 0 or very low value
        pipeline_wpm = verbal.get("wpm", 0.0)
        if metrics.wpm > 0 and (pipeline_wpm == 0 or metrics.wpm > pipeline_wpm * 1.5):
            logger.info(f"Overriding WPM: pipeline={pipeline_wpm:.1f} -> incremental={metrics.wpm:.1f}")
            verbal["wpm"] = round(metrics.wpm, 1)
            
            # Recalculate articulation rate if we override WPM
            syll_per_word_es = 2.3
            verbal["articulation_rate_sps"] = round((metrics.wpm * syll_per_word_es) / 60.0, 2)
        
        # Use incremental fillers if available
        if metrics.fillers_per_min > 0:
            logger.info(f"Overriding fillers: pipeline={verbal.get('fillers_per_min', 0):.2f} -> incremental={metrics.fillers_per_min:.2f}")
            verbal["fillers_per_min"] = round(metrics.fillers_per_min, 2)
        
        # Override nonverbal metrics with incremental data
        nonverbal = result.setdefault("nonverbal", {})
        
        if metrics.gesture_rate > 0:
            logger.info(f"Overriding gesture_rate: pipeline={nonverbal.get('gesture_rate_per_min', 0):.2f} -> incremental={metrics.gesture_rate:.2f}")
            nonverbal["gesture_rate_per_min"] = round(metrics.gesture_rate, 2)
        
        if metrics.expression_variability > 0:
            logger.info(f"Overriding expression_variability: pipeline={nonverbal.get('expression_variability', 0):.2f} -> incremental={metrics.expression_variability:.2f}")
            nonverbal["expression_variability"] = round(metrics.expression_variability, 2)

        # Recalculate scores with updated metrics
        from video.scoring import compute_scores
        
        analysis_data = {
            "verbal": verbal,
            "prosody": result.get("prosody", {}),
            "nonverbal": nonverbal
        }
        
        logger.info("Recalculating scores with enhanced metrics")
        recalculated_scores = compute_scores(analysis_data)
        result["scores"] = recalculated_scores
        logger.info(f"Scores recalculated: {recalculated_scores}")

        return result
    
    def _create_fallback_result(self) -> Dict[str, Any]:
        """Create result from incremental data as fallback."""
        state_summary = self._coordinator.get_state_summary()
        metrics = self._coordinator.metrics_analyzer.get_metrics()
        
        return {
            "status": "completed",
            "quality": {
                "frames_analyzed": state_summary["frames_received"],
                "audio_analyzed_sec": state_summary["audio_processed_sec"],
                "analysis_ms": int(state_summary["session_duration"] * 1000),
                "audio_available": True,
                "debug": {
                    "fallback_mode": True,
                    "source": "incremental_data"
                }
            },
            "media": {
                "frames_total": state_summary["frames_received"],
                "frames_with_face": state_summary["frames_received"],
                "fps": 30,
                "duration_sec": state_summary["session_duration"]
            },
            "scores": {
                "fluency": min(100, int(max(0, 100 - metrics.fillers_per_min * 20))),
                "clarity": 75,
                "pace": min(100, int(max(0, 100 - abs(metrics.wpm - 150) / 2))),
                "engagement": min(100, int(max(0, metrics.gesture_rate * 25 + metrics.expression_variability * 50)))
            },
            "verbal": {
                "wpm": round(metrics.wpm, 1),
                "fillers_per_min": round(metrics.fillers_per_min, 2),
                "filler_counts": {}
            },
            "events": [],
            "timestamp": time.time()
        }
    
    @property
    def frame_count(self) -> int:
        """Get frame count (for compatibility)."""
        return self._coordinator.get_frame_count()
    
    @property
    def streaming_active(self) -> bool:
        """Get streaming status (for compatibility)."""
        return self._coordinator.streaming_active
    
    @property
    def frame_buffer(self):
        """Get frame buffer (for compatibility)."""
        return self._coordinator.video_manager.frame_buffer
    
    def close(self) -> None:
        """Clean up resources."""
        self._coordinator.close()
        logger.info("IncrementalOratorySession closed")


async def handle_incremental_oratory_feedback(
    websocket: WebSocket,
    buffer_size: int = None,
    width: int = None,
    height: int = None,
    incremental_interval: int = None
) -> None:
    """
    Handle WebSocket connection for incremental oratory feedback.
    
    Args:
        websocket: WebSocket connection
        buffer_size: Size of the frame buffer (uses config default if None)
        width: Width of video frames (uses config default if None)
        height: Height of video frames (uses config default if None)
        incremental_interval: Process incrementally every N frames (uses config default if None)
    """
    await websocket.accept()

    # Get user_id from query parameters
    user_id = websocket.query_params.get("user_id")
    if user_id is None:
        await websocket.close(code=1008, reason="user_id is required in query params")
        return

    # Use configuration defaults if not provided
    buffer_size = buffer_size or incremental_config.frame_buffer_size
    width = width or incremental_config.default_width
    height = height or incremental_config.default_height
    incremental_interval = incremental_interval or incremental_config.processing_interval

    # Initialize session
    session = None

    try:
        session = IncrementalOratorySession(
            buffer_size=buffer_size,
            width=width,
            height=height,
            processing_interval=incremental_interval
        )
        
        # Send initial connection message
        await websocket.send_json({
            "status": "connected",
            "message": "Conexión establecida. Envía frames de video para análisis incremental.",
            "timestamp": time.time()
        })
        
        # Track frames received for incremental processing
        frames_since_last_process = 0
        last_activity_time = time.time()
        inactivity_timeout = incremental_config.inactivity_timeout_sec
        
        while True:
            try:
                # Use a short timeout to detect end of stream
                message = await asyncio.wait_for(
                    websocket.receive(), 
                    timeout=inactivity_timeout
                )
                last_activity_time = time.time()
                
                # Handle different message types
                if "bytes" in message:
                    data = message["bytes"]
                    
                    # Check prefix for audio chunks
                    if data.startswith(b'AUD'):
                        session.add_audio(data[3:])
                        continue
                    
                    # Process video frame
                    if session.add_frame(data):
                        frames_since_last_process += 1
                        
                        # Perform incremental processing at regular intervals
                        if frames_since_last_process >= incremental_interval:
                            # Send progress update
                            await websocket.send_json({
                                "status": "processing",
                                "message": f"Procesando incrementalmente ({session.frame_count} frames)",
                                "frames_processed": session.frame_count,
                                "timestamp": time.time()
                            })
                            
                            # Run incremental processing
                            result = await session.process_incremental()
                            
                            # Send detailed incremental update with metrics
                            if "incremental_metrics" in result:
                                await websocket.send_json({
                                    "status": "incremental_update",
                                    "frames_processed": session.frame_count,
                                    "metrics": result["incremental_metrics"],
                                    "confidence": result.get("confidence", 0.5),
                                    "timestamp": time.time()
                                })
                                
                                # Check if we have any recent fillers to report
                                if "recent_fillers" in result and result["recent_fillers"]:
                                    await websocket.send_json({
                                        "status": "filler_detected",
                                        "fillers": result["recent_fillers"],
                                        "timestamp": time.time()
                                    })
                            else:
                                # Send simplified update (backward compatibility)
                                await websocket.send_json({
                                    "status": "incremental_update",
                                    "frames_processed": session.frame_count,
                                    "timestamp": time.time()
                                })
                            
                            frames_since_last_process = 0
                            
                elif "text" in message:
                    # Handle text commands
                    try:
                        cmd = json.loads(message["text"])
                        
                        if cmd.get("action") == "end_stream":
                            # End the stream and process final result
                            await websocket.send_json({
                                "status": "processing",
                                "message": "Finalizando stream y generando análisis final...",
                                "timestamp": time.time()
                            })
                            
                            # Signal stream end
                            session.end_stream()
                            
                            # Generate final feedback
                            result = await session.generate_final_feedback()
                            
                            # Verificar que el resultado no sea None
                            if result is None:
                                logger.error("generate_final_feedback returned None result")
                                result = {
                                    "status": "error",
                                    "error": "Failed to generate analysis",
                                    "timestamp": time.time()
                                }
                            
                            # Send the result to the REST endpoint and wait for completion
                            try:
                                await send_analysis_result(user_id, result)
                                # Send final status to client
                                await websocket.send_json({
                                    "status": "completed",
                                    "message": "Análisis completado exitosamente",
                                    "timestamp": time.time()
                                })
                            except Exception as send_error:
                                logger.error(f"Failed to save analysis to backend: {send_error}")
                                await websocket.send_json({
                                    "status": "completed_with_warning",
                                    "message": "Análisis completado pero no se pudo guardar en el servidor",
                                    "error": str(send_error),
                                    "timestamp": time.time()
                                })
                            
                            # Close websocket after sending final result
                            await websocket.close(code=1000, reason="Analysis completed")
                            break
                            
                        elif cmd.get("action") == "status":
                            # Return current status
                            await websocket.send_json({
                                "status": "info",
                                "frames_processed": session.frame_count,
                                "buffer_size": len(session.frame_buffer),
                                "streaming_active": session.streaming_active,
                                "timestamp": time.time()
                            })
                            
                    except Exception as e:
                        logger.error(f"Error processing command: {e}")
                        await websocket.send_json({
                            "status": "error",
                            "error": f"Error processing command: {str(e)}",
                            "timestamp": time.time()
                        })
                        
            except asyncio.TimeoutError:
                # Check for inactivity timeout
                if time.time() - last_activity_time > inactivity_timeout:
                    logger.info("Inactivity timeout detected, ending stream")
                    
                    # Send notification to client
                    await websocket.send_json({
                        "status": "processing",
                        "message": "Finalizando stream por inactividad...",
                        "timestamp": time.time()
                    })
                    
                    if session.frame_count > 0:
                        # End stream and generate final results
                        session.end_stream()
                        result = await session.generate_final_feedback()
                        
                        # Verificar que el resultado no sea None
                        if result is None:
                            logger.error("generate_final_feedback returned None result en timeout handler")
                            result = {
                                "status": "error",
                                "error": "Failed to generate analysis after timeout",
                                "timestamp": time.time()
                            }
                            
                        # Send to REST endpoint and wait for completion
                        try:
                            await send_analysis_result(user_id, result)
                            # Notify client
                            await websocket.send_json({
                                "status": "completed",
                                "message": "Análisis completado por inactividad",
                                "timestamp": time.time()
                            })
                        except Exception as send_error:
                            logger.error(f"Failed to save analysis to backend after timeout: {send_error}")
                            await websocket.send_json({
                                "status": "completed_with_warning",
                                "message": "Análisis completado por inactividad pero no se pudo guardar",
                                "timestamp": time.time()
                            })
                    else:
                        await websocket.send_json({
                            "status": "closed",
                            "message": "Conexión cerrada por inactividad sin frames recibidos",
                            "timestamp": time.time()
                        })
                        
                    await websocket.close(code=1000, reason="Inactivity timeout")
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected (session exists: {session is not None})")
        
        # If we received frames, complete the analysis
        if ses<sion:
            logger.info(f"Session frame_count: {session.frame_count}")
            
        if session and session.frame_count > 0:
            try:
                logger.info(f"Starting final analysis for user {user_id} with {session.frame_count} frames")
                session.end_stream()
                logger.info("Stream ended, generating final feedback...")
                result = await session.generate_final_feedback()
                logger.info("Final feedback generated successfully")
                
                # Verificar que el resultado no sea None
                if result is None:
                    logger.error("generate_final_feedback returned None result after disconnect")
                    result = {
                        "status": "error",
                        "error": "Failed to generate analysis after disconnect",
                        "timestamp": time.time()
                    }
                
                try:
                    logger.info(f"Sending analysis result to backend for user {user_id}")
                    await send_analysis_result(user_id, result)
                    logger.info(f"Analysis completed and saved after disconnect for user {user_id}")
                except Exception as send_error:
                    logger.error(f"Failed to save analysis after disconnect: {send_error}")
            except Exception as e:
                logger.error(f"Failed to complete analysis after disconnect: {e}", exc_info=True)
        else:
            if session:
                logger.warning(f"No frames received (frame_count={session.frame_count}), skipping final analysis")
            else:
                logger.warning("Session is None, skipping final analysis")
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            if websocket.client_state.name == "CONNECTED":
                await websocket.send_json({
                    "status": "error",
                    "error": f"Error: {str(e)}",
                    "timestamp": time.time()
                })
                await websocket.close(code=1011, reason=str(e))
        except Exception:
            logger.error("Failed to send error message")
            
    finally:
        # Clean up resources
        if session:
            try:
                session.close()
                logger.info("Session resources cleaned up")
            except Exception as cleanup_error:
                logger.error(f"Error during session cleanup: {cleanup_error}")
