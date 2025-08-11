 # audio/model_loader.py
"""Handles the loading of speech and audio classification models."""

import csv
import io
import logging
import os  # Import the os module
from pathlib import Path  # Use pathlib for modern path handling
from typing import Any, List, Optional, Tuple

import requests

# --- Third-party imports with safe fallbacks ---
try:
    import vosk
except ImportError:
    vosk = None

try:
    import tensorflow_hub as hub
except ImportError:
    hub = None

# --- Local imports ---
from . import constants

logger = logging.getLogger(__name__)


class VoskModelLoader:
    """A utility to safely load a Vosk model."""

    def __init__(self, lang: str = "es", model_path: Optional[str] = None):
        """
        Initializes the model loader.

        Args:
            lang: The language of the model to load (e.g., "es").
            model_path: An optional direct path to a Vosk model directory.
        """
        self.lang = lang
        self.model_path = model_path

    def load(self) -> Optional['vosk.Model']:
        """
        Loads the Vosk model from a path or by default name.

        This method first checks for the model's existence to avoid
        the Vosk library's sys.exit() call on failure.

        Returns:
            An initialized vosk.Model object, or None if loading fails or
            Vosk is not installed.
        """
        if vosk is None:
            logger.warning("Vosk library not installed. Speech recognition is disabled.")
            return None

        # Determine the path to check
        path_to_check = self.model_path
        if not path_to_check:
            # Assume the model is in the project root if no path is given
            # This makes it predictable and avoids searching multiple directories.
            path_to_check = str(Path(os.getcwd()) / constants.VOSK_MODEL_NAME_ES)

        # --- Robustness Check: Verify model path exists BEFORE loading ---
        if not os.path.exists(path_to_check):
            logger.error(
                f"Vosk model directory not found at the expected path: '{path_to_check}'. "
                f"Please download the model and place it there. Transcription will be disabled."
            )
            return None

        try:
            # Now we can safely load using the verified path
            model = vosk.Model(model_path=path_to_check)
            logger.info(f"Vosk model loaded successfully from: {path_to_check}")
            return model
        except Exception as e:
            # This will catch other errors, like corrupted model files.
            logger.error(f"Failed to load Vosk model from '{path_to_check}': {e}", exc_info=True)
            return None


class YamnetModelLoader:
    """A utility to safely load the YAMNet audio classification model."""

    def __init__(
            self,
            model_url: str = constants.YAMNET_MODEL_URL,
            class_map_url: str = constants.YAMNET_CLASS_MAP_URL,
    ):
        """
        Initializes the YAMNet model loader.

        Args:
            model_url: The URL or path to the TensorFlow Hub YAMNet model.
            class_map_url: The URL to the CSV file containing class mappings.
        """
        self.model_url = model_url
        self.class_map_url = class_map_url

    def _download_class_map(self) -> List[str]:
        """Downloads and parses the YAMNet class map CSV."""
        logger.info(f"Downloading YAMNet class map from {self.class_map_url}")
        response = requests.get(self.class_map_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        class_names = []
        # The CSV file has a header, so we skip it.
        csv_file = io.StringIO(response.text)
        reader = csv.reader(csv_file)
        next(reader)  # Skip header row
        for row in reader:
            # The class name is in the 'display_name' column (index 2)
            class_names.append(row[2])

        logger.info(f"Successfully parsed {len(class_names)} classes from YAMNet map.")
        return class_names

    def load(self) -> Tuple[Optional[Any], Optional[List[str]]]:
        """
        Loads the YAMNet model and its class map.

        Returns:
            A tuple containing (model, class_map_list).
            Returns (None, None) if loading fails or dependencies are missing.
        """
        if hub is None:
            logger.warning(
                "TensorFlow Hub not installed. "
                "YAMNet audio classification is disabled."
            )
            return None, None

        try:
            logger.info(f"Loading YAMNet model from: {self.model_url}")
            model = hub.load(self.model_url)
            class_map = self._download_class_map()
            return model, class_map

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download YAMNet class map: {e}", exc_info=True)
            return None, None
        except Exception as e:
            logger.error(f"Failed to load YAMNet model or class map: {e}", exc_info=True)
            return None, None
