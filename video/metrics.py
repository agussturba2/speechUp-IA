"""Build metrics response (migrated from video_processor.metrics)."""

from typing import Dict, List, Any
from .llm_feedback import generar_feedback_llm


def build_metrics_response(
    feedbacks: List[Dict],
    resumen_global: Dict,
    duration_sec: float,
) -> Dict:
    """Construye la respuesta JSON con todas las métricas y feedback."""

    total = resumen_global["total_frames"]
    if total == 0:
        return {
            "duration": "0s",
            "score": 0,
            "metrics": {
                k: 0
                for k in [
                    "fluency",
                    "clarity",
                    "pace",
                    "confidence",
                    "pronunciation",
                    "engagement",
                ]
            },
            "feedbacks": feedbacks,
            "resumen_global": resumen_global,
            "puntos_a_mejorar": [],
            "puntos_debiles": ["No se pudo analizar el video"],
            "custom_feedback": [
                "No se pudo analizar el video. Sube un video válido para obtener recomendaciones personalizadas."
            ],
            "custom_feedback_llm": "No se pudo analizar el video.",
        }

    # Mapeo simple basado en métricas actuales:
    fluency = resumen_global["porcentaje_frames_con_rostro"] / 100
    clarity = resumen_global["porcentaje_posturas_buenas"] / 100
    pace = 0.8  # Placeholder
    confidence = clarity
    pronunciation = 0.7  # Placeholder
    engagement = fluency
    score = round(
        (fluency + clarity + pace + confidence + pronunciation + engagement) / 6, 2
    )

    # Generate improvement and weakness points
    puntos_a_mejorar: List[str] = []
    puntos_debiles: List[str] = []

    if fluency < 0.7:
        puntos_a_mejorar.append(
            "Procura mantener tu rostro visible durante toda la presentación."
        )
    if clarity < 0.7:
        puntos_a_mejorar.append(
            "Mejora tu postura corporal para transmitir mayor claridad y seguridad."
        )
    if pace < 0.7:
        puntos_a_mejorar.append("Ajusta tu ritmo de presentación para que sea más natural.")
    if confidence < 0.7:
        puntos_a_mejorar.append("Transmite mayor confianza con tu lenguaje corporal.")
    if pronunciation < 0.7:
        puntos_a_mejorar.append("Trabaja en la pronunciación para que tu mensaje sea más claro.")
    if engagement < 0.7:
        puntos_a_mejorar.append("Busca conectar más con la audiencia mirando a cámara.")

    # Weak points (if any metric very low)
    if fluency < 0.4:
        puntos_debiles.append("Muy poca visibilidad del rostro durante la presentación.")
    if clarity < 0.4:
        puntos_debiles.append("Postura corporal deficiente la mayor parte del tiempo.")
    if confidence < 0.4:
        puntos_debiles.append("Lenguaje corporal transmite poca seguridad.")
    if engagement < 0.4:
        puntos_debiles.append("Falta de conexión visual con la audiencia.")

    # Custom feedback per buffer
    custom_feedback: List[str] = []
    if feedbacks:
        for fb in feedbacks:
            segundos = fb["frames"]
            if segundos:
                if fb["rostros_detectados"] == 0:
                    custom_feedback.append(
                        f"Entre los segundos {segundos[0]} y {segundos[-1]} no se detectó tu rostro. Intenta mantenerte dentro del encuadre."
                    )
                elif fb["posturas_buenas"] == len(segundos):
                    custom_feedback.append(
                        f"Excelente postura entre los segundos {segundos[0]} y {segundos[-1]}."
                    )
                elif fb["posturas_buenas"] == 0:
                    custom_feedback.append(
                        f"Entre los segundos {segundos[0]} y {segundos[-1]} podrías mejorar tu postura corporal."
                    )

    # Global feedback
    if fluency > 0.85 and clarity > 0.85:
        custom_feedback.append(
            "¡Excelente oratoria! Mantén esa presencia y postura durante toda la presentación."
        )
    if fluency < 0.7:
        custom_feedback.append(
            "Hubo varios momentos donde tu rostro no fue visible. Recuerda mirar a la cámara y mantenerte en cuadro."
        )
    if clarity < 0.7:
        custom_feedback.append(
            "Trabaja en mantener la cabeza erguida y los hombros alineados para mejorar tu claridad."
        )
    if not custom_feedback:
        custom_feedback.append(
            "Buen trabajo, puedes seguir perfeccionando tu presencia y postura."
        )

    metricas_dict = {
        "fluency": fluency,
        "clarity": clarity,
        "pace": pace,
        "confidence": confidence,
        "pronunciation": pronunciation,
        "engagement": engagement,
    }

    llm_feedback = generar_feedback_llm(
        metricas_dict,
        resumen_global,
        custom_feedback,
        puntos_a_mejorar,
        puntos_debiles,
    )

    return {
        "duration": f"{duration_sec:.1f}s",
        "score": score,
        "metrics": metricas_dict,
        "feedbacks": feedbacks,
        "resumen_global": resumen_global,
        "puntos_a_mejorar": puntos_a_mejorar,
        "puntos_debiles": puntos_debiles,
        "custom_feedback": custom_feedback,
        "custom_feedback_llm": llm_feedback,
    }
