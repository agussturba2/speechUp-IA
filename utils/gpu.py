"""GPU detection and setup helpers."""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, Union

import cv2

logger = logging.getLogger(__name__)

# Global flag indicating if GPU is available
GPU_AVAILABLE = False

def _setup_opencv() -> bool:
    """Return True if OpenCV can use CUDA."""
    try:
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices > 0:
            # Trigger a short info printout once for visibility
            cv2.cuda.printCudaDeviceInfo(0)
            logger.info(f"OpenCV detected {cuda_devices} CUDA device(s)")
            return True
    except Exception as e:
        logger.debug(f"OpenCV GPU not available: {e}")
    return False


def _setup_tensorflow() -> bool:
    """Configure TensorFlow GPU memory growth if TF is installed."""
    try:
        import tensorflow as tf  # noqa: WPS433 (runtime optional import)

        # Configure TensorFlow for optimal GPU usage
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        physical_devices = tf.config.list_physical_devices("GPU")
        
        if len(physical_devices) > 0:
            logger.info(f"TensorFlow detected {len(physical_devices)} GPU device(s)")
            
            # Enable memory growth to avoid allocating all GPU memory at once
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                    logger.info(f"Enabled memory growth for {device.name}")
                except Exception as mem_err:
                    logger.warning(f"Could not set memory growth for {device.name}: {mem_err}")
            
            return True
    except ImportError:
        logger.debug("TensorFlow not available")
    except Exception as e:
        logger.debug(f"TensorFlow GPU configuration error: {e}")
    return False


def _setup_mediapipe() -> bool:
    """Configure MediaPipe to use GPU acceleration."""
    try:
        # Force MediaPipe to consider GPU
        os.environ.setdefault("MEDIAPIPE_USE_GPU", "1")
        return True
    except Exception as e:
        logger.debug(f"MediaPipe GPU setup error: {e}")
        return False


def get_optimal_mediapipe_config(model_complexity: int = 1) -> Dict[str, Union[bool, int, float]]:
    """Return optimized MediaPipe configuration based on available hardware.
    
    Args:
        model_complexity: Desired model complexity (0=light, 1=medium, 2=full)
        
    Returns:
        Dict with optimized MediaPipe configuration
    """
    global GPU_AVAILABLE
    
    # Base configuration
    config = {
        "static_image_mode": False,  # Better for video streams
        "model_complexity": model_complexity,
        "smooth_landmarks": True,
        "enable_segmentation": False,  # Disable for speed
        "min_detection_confidence": 0.5,
    }
    
    # GPU optimizations
    if GPU_AVAILABLE:
        # GPU-specific settings
        # Note: MediaPipe internally checks GPU_AVAILABLE flag
        pass
        
    return config


def setup_gpu() -> bool:
    """Configure GPU backends and return availability flag."""
    global GPU_AVAILABLE
    
    opencv_gpu = _setup_opencv()
    tf_gpu = _setup_tensorflow()
    mp_gpu = _setup_mediapipe()
    
    GPU_AVAILABLE = opencv_gpu or tf_gpu
    
    if GPU_AVAILABLE:
        logger.info("GPU acceleration enabled")
    else:
        logger.info("Running in CPU-only mode")
        
    return GPU_AVAILABLE
