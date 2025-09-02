import os

import vosk

from api.formatters import logger


def start_vosk(self, lang):
    # Check for model in common locations
    try:
        # Check standard paths
        model_paths = [
            # Current directory models
            os.path.join("models", f"vosk-model-{lang}"),
            # User home directory
            os.path.join(os.path.expanduser("~"), ".cache", "vosk", f"vosk-model-{lang}"),
            # System path
            os.path.join("/usr", "local", "share", "vosk", f"vosk-model-{lang}"),
            # Just the language code as model name
            lang
        ]

        model_found = False
        for path in model_paths:
            if os.path.exists(path) and os.path.isdir(path):
                try:
                    self.vosk_model = vosk.Model(model_path=path)
                    logger.info(f"Vosk model loaded from: {path}")
                    model_found = True
                    break
                except Exception as e:
                    logger.debug(f"Could not load model from {path}: {e}")

        # If model not found, try using just the language code
        if not model_found:
            model_found = use_language_model(lang, model_found, self)

        # If still no model, use small model if available
        if not model_found:
            use_small_model(lang, self)

    except Exception as e:
        logger.warning(f"Error during model initialization: {e}")


def use_language_model(lang, model_found, self):
    logger.info(f"Attempting to load model with language code: {lang}")
    try:
        # Try directly with language code (newer Vosk versions might support this)
        self.vosk_model = vosk.Model(lang=lang)
        logger.info(f"Vosk model loaded with language code: {lang}")
        model_found = True
    except Exception as e:
        logger.warning(f"Could not load model with language code: {e}")
    return model_found


def use_small_model(lang, self):
    logger.warning(f"No model found for language: {lang}")
    # Just try initializing with 'small' as a final fallback
    try:
        self.vosk_model = vosk.Model(model_path="small")
        logger.info("Using 'small' Vosk model as fallback")
    except Exception as small_err:
        logger.warning(f"Could not load small model: {small_err}")
