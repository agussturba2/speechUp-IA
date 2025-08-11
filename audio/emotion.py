"""
Emotion detection functionality for the oratory feedback system.
Provides functions to analyze emotional aspects of speech using YAMNet model.
"""

import numpy as np
import librosa
import logging
from typing import Dict, Any, List

# Optional imports - define at module level
tf = None
hub = None

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError:
    # Already set to Noneaudio_results
    pass

logger = logging.getLogger(__name__)

class EmotionDetector:
    """Emotion detection in speech for oratory analysis."""
    
    # YAMNet audio classes relevant to speech emotions
    _SPEECH_EMOTION_MAP = {
        # Positive emotional tones
        "Laughter": "positive",
        "Chuckle, chortle": "positive",
        "Giggle": "positive",
        "Cheering": "positive",
        "Applause": "positive",
        "Children shouting": "energetic",
        
        # Neutral speech tones
        "Speech": "neutral",
        "Narration, monologue": "neutral",
        "Conversation": "neutral",
        "Male speech, man speaking": "neutral",
        "Female speech, woman speaking": "neutral",
        
        # Negative emotional tones
        "Crying, sobbing": "negative",
        "Sigh": "negative",
        "Groan": "negative",
        "Whimper": "negative",
        "Yell": "intense",
        "Children crying": "negative",
        
        # Audio quality issues
        "Inside, small room": "audio_quality",
        "Echo": "audio_quality",
        "Noise": "audio_quality",
        "Static": "audio_quality",
        "White noise": "audio_quality",
        "Pink noise": "audio_quality",
        "Environmental noise": "audio_quality",
        "Buzz": "audio_quality",
        "Hum": "audio_quality",
        "Rustle": "audio_quality",
    }
    
    def __init__(self, model_url: str = "https://tfhub.dev/google/yamnet/1", download_if_missing: bool = True):
        """
        Initialize emotion detector.
        
        Args:
            model_url: TensorFlow Hub URL for YAMNet model
            download_if_missing: Whether to download the model if not found
        """
        self.model_url = model_url
        self.yamnet_model = None
        self.class_names = None
        self.download_if_missing = download_if_missing
        
        # Lazy-load YAMNet via TF-Hub
        if hub is not None and tf is not None:
            try:
                # Try to suppress some TF warnings
                import os
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                
                # Configure TF for optimal performance
                try:
                    # Use dynamic memory allocation
                    physical_devices = tf.config.list_physical_devices('GPU')
                    if physical_devices:
                        tf.config.experimental.set_memory_growth(physical_devices[0], True)
                        logger.info("Configured TensorFlow for dynamic GPU memory allocation")
                except Exception as gpu_err:
                    logger.debug(f"GPU configuration error (not critical): {gpu_err}")
                
                # Load the model
                logger.info(f"Loading YAMNet model from {model_url}...")
                self.yamnet_model = hub.load(model_url)
                
                # Load class names
                import urllib.request
                class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
                with urllib.request.urlopen(class_map_url) as file:
                    class_map_csv = file.read().decode('utf-8')
                    self.class_names = [line.split(',')[0] for line in class_map_csv.splitlines()[1:]]
                    
                logger.info(f"YAMNet model loaded successfully with {len(self.class_names)} classes")
                
            except Exception as e:
                logger.warning(f"Could not load YAMNet: {e}")
                if download_if_missing and hub is not None:  # Only attempt if hub is available
                    logger.info("Attempting to download model...")
                    try:
                        # Use the global hub module, not a new import
                        hub.resolve(model_url)
                        logger.info("Model downloaded successfully")
                        # Try loading again
                        self.yamnet_model = hub.load(model_url)
                    except Exception as dl_err:
                        logger.warning(f"Model download failed: {dl_err}")
        else:
            logger.warning("TensorFlow/TF-Hub not available. Emotion analysis will be disabled.")
    
    def analyze(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Analyze emotions in audio using YAMNet.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary with detected emotions, scores, and audio characteristics
        """
        if self.yamnet_model is None or tf is None:
            logger.warning("No emotion detection model available")
            return {
                "emotion_top_labels": [],
                "emotion_top_scores": [],
                "emotion_categories": {},
                "dominant_emotion": "unknown",
                "audio_quality": "unknown",
                "audio_characteristics": []
            }
        
        try:    
            # Check if we have valid audio
            if len(audio) < sr * 0.5:  # Need at least 0.5s of audio
                logger.warning("Audio too short for emotion analysis")
                return {"error": "Audio too short for analysis"}
            
            # Segment audio into chunks for better analysis
            segment_duration = 3.0  # 3 seconds per segment
            segment_samples = int(segment_duration * 16000)
            
            # Resample to 16kHz for YAMNet if needed
            if sr != 16000:
                audio16 = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio16 = audio
            
            # Process in segments to handle long audio better
            all_scores = []
            for start in range(0, len(audio16), segment_samples):
                end = min(start + segment_samples, len(audio16))
                if end - start < 8000:  # Skip if less than 0.5 seconds
                    continue
                    
                # YAMNet expects mono waveform float32
                segment = audio16[start:end]
                wav = tf.convert_to_tensor(segment, dtype=tf.float32)
                
                # Analyze segment
                scores, embeddings, spectrogram = self.yamnet_model(wav)
                
                # Append segment scores for further analysis
                all_scores.append(scores.numpy())
            
            # Combine scores from all segments
            if not all_scores:
                logger.warning("No valid audio segments for analysis")
                return {"error": "No valid audio segments for analysis"}
                
            combined_scores = np.vstack(all_scores)
            mean_scores = np.mean(combined_scores, axis=0)
            
            # Get top detected sounds (not just emotions)
            top_count = min(5, len(mean_scores))
            top_indices = mean_scores.argsort()[-top_count:][::-1]
            top_labels = [self.class_names[i] if i < len(self.class_names) else f"Unknown-{i}" for i in top_indices]
            top_scores = mean_scores[top_indices].tolist()
            
            # Map to emotion categories
            emotion_data = {}
            audio_characteristics = []
            
            # Process all scores with reasonable confidence
            threshold = 0.1  # Only consider sounds with at least 10% confidence
            for i, score in enumerate(mean_scores):
                if score >= threshold and i < len(self.class_names):
                    label = self.class_names[i]
                    emotion_cat = self._SPEECH_EMOTION_MAP.get(label)
                    
                    if emotion_cat:
                        # For recognized emotion categories
                        emotion_data[emotion_cat] = max(emotion_data.get(emotion_cat, 0), score)
                        
                        # Add to audio characteristics if significant
                        if score >= 0.15 and label not in audio_characteristics:
                            audio_characteristics.append(label)
            
            # Determine dominant emotion and audio quality
            dominant_emotion = "neutral"  # Default
            if emotion_data:
                dominant_item = max(emotion_data.items(), key=lambda x: x[1] if x[0] != "audio_quality" else 0)
                dominant_emotion = dominant_item[0] if dominant_item[0] != "audio_quality" else "neutral"
            
            # Check for audio quality issues
            audio_quality = "good"  # Default
            if "audio_quality" in emotion_data and emotion_data["audio_quality"] > 0.3:
                audio_quality = "poor"
            elif "audio_quality" in emotion_data and emotion_data["audio_quality"] > 0.15:
                audio_quality = "fair"
            
            # Build comprehensive result
            result = {
                "emotion_top_labels": top_labels,
                "emotion_top_scores": top_scores,
                "emotion_categories": emotion_data,
                "dominant_emotion": dominant_emotion,
                "audio_quality": audio_quality,
                "audio_characteristics": audio_characteristics[:5]  # Limit to top 5
            }
            
            logger.info(f"Emotion analysis complete: dominant={dominant_emotion}, quality={audio_quality}")
            return result
            
        except Exception as e:
            logger.error(f"Error during emotion analysis: {str(e)}")
            return {
                "error": str(e),
                "emotion_top_labels": [],
                "dominant_emotion": "unknown",
                "audio_quality": "unknown"
            }
