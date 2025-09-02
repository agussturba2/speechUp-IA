from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def format_velocidad_feedback(audio_analysis: Dict[str, Any], resumen: Dict[str, Any], analisis_habla: Dict[str, Any]) -> None:
    wpm = audio_analysis.get("words_per_minute")
    if wpm is None:
        return

    analisis_habla["velocidad"]["valor"] = wpm
    if wpm < 100:
        analisis_habla["velocidad"]["evaluacion"] = "Tu ritmo de habla es lento"
        analisis_habla["velocidad"]["recomendacion"] = "Intenta aumentar ligeramente tu velocidad para mantener mejor el interés de la audiencia"
        resumen["areas_de_mejora"].append(f"⚠️ Ritmo de habla demasiado lento ({int(wpm)} palabras/min)")
    elif wpm > 160:
        analisis_habla["velocidad"]["evaluacion"] = "Tu ritmo de habla es bastante rápido"
        analisis_habla["velocidad"]["recomendacion"] = "Considera reducir ligeramente la velocidad para mejorar la claridad"
        resumen["areas_de_mejora"].append(f"⚠️ Ritmo de habla demasiado rápido ({int(wpm)} palabras/min)")
    else:
        analisis_habla["velocidad"]["evaluacion"] = "Tu ritmo de habla es adecuado"
        analisis_habla["velocidad"]["recomendacion"] = "Mantén este ritmo cómodo de habla"
        resumen["aspectos_destacados"].append(f"✅ Excelente ritmo de habla ({int(wpm)} palabras/min)")

def format_emocion_feedback(audio_analysis: Dict[str, Any], resumen: Dict[str, Any], analisis_habla: Dict[str, Any]) -> None:
    emotion = audio_analysis.get("dominant_emotion")
    if not emotion:
        return

    emotion_map = {
        "neutral": "neutral",
        "positive": "positiva",
        "happy": "feliz",
        "enthusiastic": "entusiasta",
        "sad": "triste",
        "angry": "enojado/a",
        "fear": "temeroso/a"
    }
    emotion_es = emotion_map.get(emotion, emotion)
    analisis_habla["emocion"]["dominante"] = emotion_es

    if emotion == "neutral":
        analisis_habla["emocion"]["evaluacion"] = "Tu tono es principalmente neutral"
        analisis_habla["emocion"]["recomendacion"] = "Intenta añadir más variedad vocal para captar mejor a tu audiencia"
        resumen["areas_de_mejora"].append("⚠️ Añadir más variedad y emoción en la voz")
    elif emotion in ["positive", "happy", "enthusiastic"]:
        analisis_habla["emocion"]["evaluacion"] = "Tu tono transmite entusiasmo"
        analisis_habla["emocion"]["recomendacion"] = "Tu tono positivo funciona bien para mantener el compromiso de la audiencia"
        resumen["aspectos_destacados"].append(f"✅ Buen entusiasmo vocal (tono {emotion_es})")

def format_pausas_feedback(audio_analysis: Dict[str, Any], resumen: Dict[str, Any], analisis_habla: Dict[str, Any]) -> None:
    pause_freq = audio_analysis.get("pause_frequency")
    pause_length = audio_analysis.get("avg_pause_length", 0)
    if pause_freq is None:
        return

    analisis_habla["pausas"]["frecuencia"] = pause_freq
    analisis_habla["pausas"]["longitud_media"] = pause_length

    if pause_freq < 5 or pause_length < 0.3:
        analisis_habla["pausas"]["evaluacion"] = "Tienes pocas pausas en tu discurso"
        analisis_habla["pausas"]["recomendacion"] = "Incluye más pausas estratégicas para enfatizar puntos importantes"
        resumen["areas_de_mejora"].append(f"⚠️ Incluir más pausas estratégicas (frecuencia: {round(pause_freq,1)}/min)")
    elif pause_freq > 15 or pause_length > 2:
        analisis_habla["pausas"]["evaluacion"] = "Tu discurso contiene muchas pausas"
        analisis_habla["pausas"]["recomendacion"] = "Intenta reducir el número de pausas prolongadas"
        resumen["areas_de_mejora"].append(f"⚠️ Reducir pausas excesivas (frecuencia: {round(pause_freq,1)}/min)")
    else:
        analisis_habla["pausas"]["evaluacion"] = "Tu uso de pausas es apropiado"
        analisis_habla["pausas"]["recomendacion"] = "Continúa usando pausas de manera efectiva"
        resumen["aspectos_destacados"].append(f"✅ Buen uso de pausas (frecuencia: {round(pause_freq,1)}/min)")

def format_muletillas_feedback(audio_analysis: Dict[str, Any], resumen: Dict[str, Any], analisis_habla: Dict[str, Any]) -> None:
    fillers_total = audio_analysis.get("fillers_total", 0)
    transcript = audio_analysis.get("transcript", {})
    total_words = 100  # fallback

    if isinstance(transcript, dict):
        if "total_words" in transcript:
            total_words = transcript["total_words"]
        elif "words" in transcript and isinstance(transcript["words"], list):
            total_words = len(transcript["words"])

    filler_percentage = (fillers_total / total_words * 100) if total_words > 0 else 0

    analisis_habla["muletillas"]["cantidad"] = fillers_total
    analisis_habla["muletillas"]["porcentaje"] = round(filler_percentage, 1)

    if filler_percentage > 5:
        analisis_habla["muletillas"]["evaluacion"] = "Tu discurso contiene muchas muletillas"
        analisis_habla["muletillas"]["recomendacion"] = "Practica reducir muletillas como 'este', 'eh', 'o sea', 'pues'"
        resumen["areas_de_mejora"].append(f"⚠️ Reducir muletillas ({fillers_total} detectadas, {round(filler_percentage,1)}% del discurso)")
    elif filler_percentage > 2:
        analisis_habla["muletillas"]["evaluacion"] = "Tu discurso contiene algunas muletillas"
        analisis_habla["muletillas"]["recomendacion"] = "Ten cuidado con el uso ocasional de muletillas"
        resumen["areas_de_mejora"].append(f"⚠️ Reducir muletillas ocasionales ({fillers_total} detectadas)")
    else:
        analisis_habla["muletillas"]["evaluacion"] = "Utilizas muy pocas muletillas"
        analisis_habla["muletillas"]["recomendacion"] = "Excelente control de la claridad del discurso"
        resumen["aspectos_destacados"].append(f"✅ Uso mínimo de muletillas ({fillers_total} detectadas)")

def format_audio_quality_feedback(audio_analysis: Dict[str, Any], detalles_tecnicos: Dict[str, Any]) -> None:
    audio_quality_map = {
        "good": "buena",
        "poor": "deficiente",
        "fair": "aceptable",
        "excellent": "excelente",
        "bad": "mala"
    }
    audio_quality = audio_analysis.get("audio_quality")
    if audio_quality:
        detalles_tecnicos["calidad_audio"] = audio_quality_map.get(audio_quality, audio_quality)

def format_errores_feedback(results: Dict[str, Any], formatted: Dict[str, Any]) -> None:
    if "video_analysis_error" in results:
        formatted["errores"].append({
            "componente": "analisis_video",
            "mensaje": results["video_analysis_error"],
            "critico": True
        })
    if "audio_analysis_error" in results:
        formatted["errores"].append({
            "componente": "analisis_audio",
            "mensaje": results["audio_analysis_error"],
            "critico": True
        })

def calculate_puntuacion_global(formatted: Dict[str, Any]) -> None:
    score_components = []

    # Velocidad de habla
    wpm = formatted["analisis_habla"]["velocidad"].get("valor")
    if wpm is not None:
        if 120 <= wpm <= 150:
            rate_score = 100
        elif 100 <= wpm < 120 or 150 < wpm <= 170:
            rate_score = 80
        elif 80 <= wpm < 100 or 170 < wpm <= 190:
            rate_score = 60
        else:
            rate_score = 40
        score_components.append(rate_score)

    # Muletillas
    filler_percentage = formatted["analisis_habla"]["muletillas"].get("porcentaje", 0)
    if filler_percentage < 1:
        filler_score = 100
    elif filler_percentage < 3:
        filler_score = 80
    elif filler_percentage < 5:
        filler_score = 60
    else:
        filler_score = max(20, 100 - filler_percentage * 10)
    score_components.append(filler_score)

    # Pausas
    pause_freq = formatted["analisis_habla"]["pausas"].get("frecuencia", 0)
    pause_length = formatted["analisis_habla"]["pausas"].get("longitud_media", 0)
    if pause_freq > 0:
        if 8 <= pause_freq <= 12 and 0.5 <= pause_length <= 1.5:
            pause_score = 100
        elif 5 <= pause_freq < 8 or 12 < pause_freq <= 15:
            pause_score = 80
        else:
            pause_score = 60
        score_components.append(pause_score)

    if score_components:
        formatted["resumen"]["puntuacion_global"] = round(sum(score_components) / len(score_components))

def format_comprehensive_feedback(results: Dict[str, Any], language: str = "es") -> Dict[str, Any]:
    formatted = {
        "resumen": {
            "aspectos_destacados": [],
            "areas_de_mejora": [],
            "puntuacion_global": 0
        },
        "analisis_habla": {
            "velocidad": {"valor": None, "evaluacion": "", "recomendacion": ""},
            "claridad": {"valor": None, "evaluacion": "", "recomendacion": ""},
            "emocion": {"dominante": "", "variedad": 0, "evaluacion": "", "recomendacion": ""},
            "pausas": {"frecuencia": 0, "longitud_media": 0, "evaluacion": "", "recomendacion": ""},
            "muletillas": {"cantidad": 0, "porcentaje": 0, "evaluacion": "", "recomendacion": ""}
        },
        "analisis_presentacion": {
            "visibilidad_rostro": {"porcentaje": 0, "evaluacion": "", "recomendacion": ""},
            "postura": {"calidad": 0, "evaluacion": "", "recomendacion": ""}
        },
        "transcripcion": {"texto": "", "cantidad_palabras": 0, "duracion_minutos": 0},
        "detalles_tecnicos": {"tiempo_procesamiento": 0, "gpu_utilizada": False, "calidad_video": "", "calidad_audio": ""},
        "errores": []
    }

    format_errores_feedback(results, formatted)

    if "total_processing_time_sec" in results:
        formatted["detalles_tecnicos"]["tiempo_procesamiento"] = results["total_processing_time_sec"]

    perf = results.get("performance", {})
    formatted["detalles_tecnicos"]["gpu_utilizada"] = perf.get("gpu_used", False)
    if "processing_ratio_percent" in perf:
        formatted["detalles_tecnicos"]["porcentaje_frames_procesados"] = perf["processing_ratio_percent"]
    if "frames_per_second" in perf:
        formatted["detalles_tecnicos"]["frames_por_segundo"] = perf["frames_per_second"]

    audio_analysis = results.get("audio_analysis", {})
    if audio_analysis:
        format_velocidad_feedback(audio_analysis, formatted["resumen"], formatted["analisis_habla"])
        format_emocion_feedback(audio_analysis, formatted["resumen"], formatted["analisis_habla"])
        format_pausas_feedback(audio_analysis, formatted["resumen"], formatted["analisis_habla"])
        format_muletillas_feedback(audio_analysis, formatted["resumen"], formatted["analisis_habla"])
        format_audio_quality_feedback(audio_analysis, formatted["detalles_tecnicos"])

        # Transcripción
        transcript = audio_analysis.get("transcript", {})
        if isinstance(transcript, dict) and "text" in transcript:
            formatted["transcripcion"]["texto"] = transcript["text"]

        total_words = 0
        if isinstance(transcript, dict):
            if "total_words" in transcript:
                total_words = transcript["total_words"]
            elif "words" in transcript and isinstance(transcript["words"], list):
                total_words = len(transcript["words"])
        formatted["transcripcion"]["cantidad_palabras"] = total_words

        if "duration_seconds" in audio_analysis:
            formatted["transcripcion"]["duracion_minutos"] = round(audio_analysis["duration_seconds"] / 60, 2)

    calculate_puntuacion_global(formatted)

    return formatted

