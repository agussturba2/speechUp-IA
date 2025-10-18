"""
WebSocket handlers for real-time oratory analysis and feedback.
"""

import logging
import os
import time
import tempfile
import json
from typing import Dict, Any, List
import cv2
from fastapi import WebSocket, WebSocketDisconnect
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import asyncio
import httpx

from video.pipeline import run_analysis_pipeline
from video.metrics import build_metrics_response
from api.schemas.analysis_json import AnalysisJSON
from video.realtime import decode_frame_data

logger = logging.getLogger(__name__)

# Función helper para procesar resultados del pipeline (similar a la clase REST)
def process_pipeline_results(proc: Dict[str, Any], analysis_time: float) -> Dict[str, Any]:
    """
    Process pipeline results into standardized format.
    """
    # Normalize fps to integer
    if isinstance(proc.get("fps"), float):
        proc["fps"] = int(round(proc["fps"]))

    # Normalize events list structure
    events = proc.get("events", [])
    normalized_events = []
    for e in events:
        normalized_event = {
            "t": e.get("t") or e.get("time_sec") or e.get("timestamp", 0.0),
            "kind": e.get("kind") or e.get("type", "gesture"),
            "label": e.get("label"),
            "duration": e.get("duration")
        }
        if "score" in e:
            normalized_event["score"] = e["score"]
        if "confidence" in e:
            normalized_event["confidence"] = e["confidence"]
        normalized_events.append(normalized_event)

    # Build media dict with derived nonverbal metrics
    media = {
        "frames_total":       proc.get("frames_total", 0),
        "frames_with_face":   proc.get("frames_with_face", 0),
        "fps":                proc.get("fps", 0.0),
        "duration_sec":       proc.get("duration_sec", 0.0),
        "dropped_frames_pct": proc.get("dropped_frames_pct", 0.0),
        "gaze_screen_pct":    proc.get("gaze_screen_pct", 0.0),
        "head_stability":     proc.get("head_stability", 0.0),
        "gesture_amplitude":  proc.get("gesture_amplitude", 0.0),
        "posture_openness":   proc.get("posture_openness", 0.0),
        "expression_variability": proc.get("expression_variability", 0.0),
        "engagement":         proc.get("engagement", 0.0),
        "gesture_events":     proc.get("gesture_events", 0),
    }

    gesture_events = media["gesture_events"]

    # Get audio availability and verbal/prosody data from pipeline
    pipeline_audio_available = proc.get("audio_available", False)
    pipeline_verbal = proc.get("verbal", {})
    pipeline_prosody = proc.get("prosody", {})



    if "filler_counts" in pipeline_verbal and not isinstance(pipeline_verbal["filler_counts"], dict):
        pipeline_verbal["filler_counts"] = {}

    # Asegurar que todos los campos numéricos tengan valores por defecto
    verbal_defaults = {
        "wpm": 0.0,
        "articulation_rate_sps": 0.0,
        "fillers_per_min": 0.0,
        "avg_pause_sec": 0.0,
        "pause_rate_per_min": 0.0,
        "pronunciation_score": 0.0,
        "stt_confidence": 0.0
    }

    for key, default_value in verbal_defaults.items():
        if key not in pipeline_verbal or not isinstance(pipeline_verbal[key], (int, float)):
            pipeline_verbal[key] = default_value

    # Asegurar que los campos de prosody tengan valores por defecto
    prosody_defaults = {
        "pitch_mean_hz": 0.0,
        "pitch_range_semitones": 0.0,
        "pitch_cv": 0.0,
        "energy_cv": 0.0,
        "rhythm_consistency": 0.0
    }

    for key, default_value in prosody_defaults.items():
        if key not in pipeline_prosody or not isinstance(pipeline_prosody[key], (int, float)):
            pipeline_prosody[key] = default_value

    result = build_metrics_response(
        media=media,
        events=normalized_events,
        analysis_ms=0,
        verbal=pipeline_verbal,
        prosody=pipeline_prosody,
        audio_available=pipeline_audio_available,
    )
    result.setdefault("media", {}).setdefault("lang", "es-AR")

    # Inject pipeline values if present and > 0
    if proc.get("fps", 0) > 0:
        result["media"]["fps"] = proc["fps"]
    if proc.get("duration_sec", 0) > 0:
        result["media"]["duration_sec"] = proc["duration_sec"]
    if proc.get("frames_total", 0) > 0:
        result.setdefault("quality", {})["frames_analyzed"] = proc["frames_total"]

    # Set analysis_ms
    elapsed_ms = int(analysis_time * 1000)
    q = result.setdefault("quality", {})
    q["analysis_ms"] = elapsed_ms
    q.setdefault("audio_available", pipeline_audio_available)

    # Add ASR debug info if enabled and available
    if os.getenv("SPEECHUP_DEBUG_ASR", "0") == "1" and pipeline_verbal:
        asr_debug = pipeline_verbal.get("debug", {})
        if asr_debug:
            q.setdefault("debug", {})["asr"] = asr_debug

    # Ensure all scores are integers
    scores = result.get("scores", {})
    for k in ["fluency", "clarity", "delivery_confidence", "pronunciation", "pace", "engagement"]:
        if k in scores:
            scores[k] = int(round(scores[k]))
    result["scores"] = scores

    logger.info(f"Propagated fps={result['media'].get('fps')}, duration_sec={result['media'].get('duration_sec')}, frames_analyzed={result['quality'].get('frames_analyzed')}, scores={result['scores']}")

    if isinstance(result["media"].get("fps"), float):
        result["media"]["fps"] = int(round(result["media"]["fps"]))

    # Asegurar que todos los campos que deben ser arrays/listas estén en el formato correcto
    if "events" not in result or not isinstance(result["events"], list):
        result["events"] = []

    if "recommendations" not in result or not isinstance(result["recommendations"], list):
        result["recommendations"] = []

    if "lexical" not in result:
        result["lexical"] = {}

    if "keywords" not in result["lexical"] or not isinstance(result["lexical"]["keywords"], list):
        result["lexical"]["keywords"] = []


    result["long_pauses"] = []
    try:
        validated = AnalysisJSON.model_validate(result)
        return validated.model_dump()
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return result

# Función para enviar resultados al endpoint REST
async def send_analysis_result(user_id: str, result: Dict[str, Any]):
    url = "http://98.91.55.213:7070/session"
    params = {"userId": user_id}
    
    logger.error(f"=== ATTEMPTING DATABASE INSERT ===")
    logger.error(f"User ID: {user_id}")
    logger.error(f"URL: {url}")
    logger.error(f"Body keys: {list(result.keys())}")
    logger.error(f"Body size: {len(str(result))} chars")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, params=params, json=result, timeout=30.0)
            response.raise_for_status()
            logger.error(f"✅ DATABASE INSERT SUCCESSFUL - Status: {response.status_code}")
            logger.info(f"Analysis result sent to {url} successfully")
    except Exception as e:
        logger.error(f"❌ DATABASE INSERT FAILED - Error: {e}")
        logger.error(f"Failed to send analysis result to {url}: {e}")

class OratorySession:
    """
    Session handler for real-time oratory analysis.
    """

    def __init__(
        self,
        buffer_size: int = 30,
        width: int = 640,
        height: int = 480,
    ):
        """
        Initialize a new oratory analysis session.
        """
        self.buffer_size = buffer_size
        self.width = width
        self.height = height

        # Initialize frame buffer
        self.frame_buffer = deque(maxlen=buffer_size)
        self.audio_buffer = bytearray()

        # Analysis state
        self.analysis_in_progress = False
        self.last_analysis_time = 0
        self.tmp_path = None

        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=1)


    def add_frame(self, frame_data: bytes) -> bool:
        """
        Add a video frame to the buffer.
        """
        frame = decode_frame_data(frame_data, self.width, self.height)
        if frame is None:
            logger.warning("Failed to decode frame data")
            return False

        self.frame_buffer.append(frame)
        return True

    def add_audio(self, audio_data: bytes) -> None:
        """
        Add audio data to the buffer.
        """
        self.audio_buffer.extend(audio_data)

    async def generate_feedback(self) -> Dict[str, Any]:
        """
        Generate feedback from current buffers.
        """
        # Skip if buffer is empty or analysis already in progress
        if not self.frame_buffer:
            return {"status": "buffering", "buffer_size": len(self.frame_buffer)}

        if self.analysis_in_progress:
            return {"status": "analyzing", "message": "Analysis already in progress"}

        self.analysis_in_progress = True
        t0 = time.perf_counter()

        try:
            # Run analysis in thread pool to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_analysis
            )

            return result

        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "error_type": str(type(e).__name__),
                "timestamp": time.time()
            }

        finally:
            # Clean up temporary file
            self._cleanup_temp_file()
            self.analysis_in_progress = False
            self.last_analysis_time = time.perf_counter()

    def _run_analysis(self) -> Dict[str, Any]:
        """
        Run analysis pipeline on current buffer.
        """
        if not self.frame_buffer:
            raise ValueError("No frames in buffer")

        # Save frames to temporary video file
        self.tmp_path = self._save_frames_to_temp()

        # Run analysis pipeline and measure time
        t0 = time.perf_counter()
        proc = run_analysis_pipeline(self.tmp_path)
        analysis_time = time.perf_counter() - t0

        # Process results using the helper function
        return process_pipeline_results(proc, analysis_time)

    def _save_frames_to_temp(self) -> str:
        """
        Save frames to a temporary video file.
        """
        # Implementation details for saving frames to a temporary file
        # ... (código existente)

    def _cleanup_temp_file(self) -> None:
        """
        Clean up temporary file.
        """
        # ... (código existente)

    def close(self) -> None:
        """
        Clean up resources.
        """
        self._cleanup_temp_file()
        self.frame_buffer.clear()
        self.audio_buffer = bytearray()
        self.executor.shutdown(wait=False)


async def handle_oratory_feedback(
    websocket: WebSocket,
    buffer_size: int = 30,
    width: int = 640,
    height: int = 480,
    analysis_interval: float = 5.0
):
    """
    Handle WebSocket connection for oratory feedback.
    """
    await websocket.accept()

    # Get user_id from query parameters
    user_id = websocket.query_params.get("user_id")
    if user_id is None:
        await websocket.close(code=1008, reason="user_id is required in query params")
        return

    # Initialize session
    session = None

    try:
        session = OratorySession(
            buffer_size=buffer_size,
            width=width,
            height=height,
        )

        # Timing control
        last_analysis_time = 0

        # Send initial connection message
        await websocket.send_json({
            "status": "connected",
            "message": "Conexión establecida. Envía frames de video para análisis.",
            "timestamp": time.time()
        })

        while True:
            try:
                # Receive message with timeout and connection check
                message = await asyncio.wait_for(websocket.receive(), timeout=300.0)  # 5 minutos timeout

                # Handle different message types
                if "bytes" in message:
                    data = message["bytes"]

                    # Check prefix for audio chunks
                    if data.startswith(b'AUD'):
                        session.add_audio(data[3:])
                        continue

                    # Check if it's a complete video file
                    if data.startswith(b'VID'):
                        # Notify client that we're processing a complete video
                        await websocket.send_json({
                            "status": "in_progress",
                            "message": "Procesando video completo...",
                            "timestamp": time.time()
                        })

                        # Save video to temporary file
                        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
                        os.close(fd)

                        try:
                            # Write video data to file
                            with open(tmp_path, "wb") as f:
                                f.write(data[3:])  # Skip the 'VID' prefix

                            # Process video in chunks but only send status updates
                            cap = cv2.VideoCapture(tmp_path)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)

                            # Send initial info
                            await websocket.send_json({
                                "status": "in_progress",
                                "message": f"Procesando video de {total_frames} frames...",
                                "timestamp": time.time()
                            })

                            # Process video in chunks
                            chunk_size = min(30, total_frames)
                            chunks = total_frames // chunk_size

                            for i in range(chunks):
                                # Clear previous frames
                                session.frame_buffer.clear()

                                # Read chunk_size frames
                                for _ in range(chunk_size):
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    session.frame_buffer.append(frame)

                                # Generate feedback for this chunk but don't send detailed results
                                if session.frame_buffer:
                                    # Just update progress
                                    progress = min(100, int(100 * (i + 1) / chunks))

                                    # Only send progress updates every few chunks to reduce traffic
                                    if i % 5 == 0 or i == chunks - 1:
                                        await websocket.send_json({
                                            "status": "in_progress",
                                            "message": f"Procesando... {progress}% completado",
                                            "progress": progress,
                                            "timestamp": time.time()
                                        })

                            # Final analysis with all available data
                            cap.release()

                            # Run the full pipeline on the complete video and measure time
                            t0_pipeline = time.perf_counter()
                            proc = run_analysis_pipeline(tmp_path)
                            analysis_time = time.perf_counter() - t0_pipeline

                            # Process the results to generate the full JSON
                            result = process_pipeline_results(proc, analysis_time)

                            # Send the result to the REST endpoint and wait for completion
                            try:
                                await send_analysis_result(user_id, result)
                                # Send final results to client
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

                        except Exception as e:
                            logger.error(f"Error processing video: {str(e)}")
                            try:
                                await websocket.send_json({
                                    "status": "error",
                                    "message": f"Error al procesar el video: {str(e)}",
                                    "timestamp": time.time()
                                })
                            except Exception:
                                logger.error("Failed to send error message to client", exc_info=True)
                        finally:
                            # Clean up
                            if os.path.exists(tmp_path):
                                try:
                                    os.remove(tmp_path)
                                except Exception as e:
                                    logger.error(f"Error removing temp file: {str(e)}")

                        continue

                    # Process regular video frame
                    if session.add_frame(data):
                        # Send buffer status periodically (not on every frame)
                        current_time = time.time()
                        buffer_full = len(session.frame_buffer) == session.buffer_size
                        time_elapsed = current_time - last_analysis_time >= analysis_interval

                        # Run analysis when buffer is full and enough time has passed
                        if buffer_full and time_elapsed and not session.analysis_in_progress:
                            # Notify client that analysis is starting
                            await websocket.send_json({
                                "status": "in_progress",
                                "message": "Analizando video...",
                                "timestamp": current_time
                            })

                            # Generate feedback
                            feedback = await session.generate_feedback()

                            # Send the result to the REST endpoint and wait for completion
                            try:
                                await send_analysis_result(user_id, feedback)
                                # Send simplified status to client
                                await websocket.send_json({
                                    "status": "completed",
                                    "message": "Análisis completado",
                                    "timestamp": time.time()
                                })
                            except Exception as send_error:
                                logger.error(f"Failed to save analysis to backend: {send_error}")
                                await websocket.send_json({
                                    "status": "completed_with_warning",
                                    "message": "Análisis completado pero no se pudo guardar",
                                    "timestamp": time.time()
                                })

                            last_analysis_time = time.time()

                        # Periodically send buffer status updates (every 30 frames)
                        elif len(session.frame_buffer) % 30 == 0:
                            await websocket.send_json({
                                "status": "buffering",
                                "buffer_size": len(session.frame_buffer),
                                "buffer_capacity": session.buffer_size,
                                "timestamp": current_time
                            })

                elif "text" in message:
                    # Handle text commands
                    try:
                        cmd = json.loads(message["text"])

                        if cmd.get("action") == "analyze" and len(session.frame_buffer) > 0:
                            # Force analysis if not already in progress
                            if not session.analysis_in_progress:
                                await websocket.send_json({
                                    "status": "in_progress",
                                    "message": "Analizando video por solicitud...",
                                    "timestamp": time.time()
                                })

                                feedback = await session.generate_feedback()

                                # Send the result to the REST endpoint and wait for completion
                                try:
                                    await send_analysis_result(user_id, feedback)
                                    await websocket.send_json({
                                        "status": "completed",
                                        "message": "Análisis completado",
                                        "timestamp": time.time()
                                    })
                                except Exception as send_error:
                                    logger.error(f"Failed to save analysis to backend: {send_error}")
                                    await websocket.send_json({
                                        "status": "completed_with_warning",
                                        "message": "Análisis completado pero no se pudo guardar",
                                        "timestamp": time.time()
                                    })

                                last_analysis_time = time.time()
                            else:
                                await websocket.send_json({
                                    "status": "busy",
                                    "message": "Análisis en progreso, espere por favor.",
                                    "timestamp": time.time()
                                })

                        elif cmd.get("action") == "clear":
                            # Clear buffer
                            session.frame_buffer.clear()
                            await websocket.send_json({
                                "status": "cleared",
                                "message": "Buffer limpiado",
                                "timestamp": time.time()
                            })

                        elif cmd.get("action") == "status":
                            # Return current status
                            await websocket.send_json({
                                "status": "info",
                                "buffer_size": len(session.frame_buffer),
                                "buffer_capacity": session.buffer_size,
                                "analysis_in_progress": session.analysis_in_progress,
                                "timestamp": time.time()
                            })

                    except Exception as e:
                        try:
                            await websocket.send_json({
                                "status": "error",
                                "error": f"Error processing command: {str(e)}",
                                "timestamp": time.time()
                            })
                        except Exception:
                            logger.error("Failed to send error message to client", exc_info=True)

            except asyncio.TimeoutError:
                # Timeout sin actividad, cerrar conexión
                logger.info("WebSocket timeout due to inactivity")
                await websocket.close(code=1000, reason="Timeout por inactividad")
                break

            except WebSocketDisconnect:
                # Cliente se desconectó normalmente
                logger.info("Client disconnected normally")
                break

            except RuntimeError as e:
                if "disconnect message has been received" in str(e):
                    # Cliente se desconectó, salir del loop
                    logger.info("Client disconnected (disconnect message received)")
                    break
                else:
                    # Otro error RuntimeError, relanzar
                    raise

    except WebSocketDisconnect:
        logger.info("Client disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            # Verificar si la conexión todavía está abierta antes de enviar mensajes
            if websocket.client_state.name == "CONNECTED":
                await websocket.send_json({
                    "status": "error",
                    "error": f"Error: {str(e)}",
                    "error_type": str(type(e).__name__),
                    "timestamp": time.time()
                })
        except Exception:
            logger.error("Failed to send error message to client", exc_info=True)

    finally:
        # Clean up resources
        if session:
            try:
                session.close()
                logger.debug("Session resources cleaned up successfully")
            except Exception as cleanup_error:
                logger.error(f"Error during session cleanup: {str(cleanup_error)}", exc_info=True)