"""LLM-based feedback generator (migrated from video_processor.llm_feedback)."""

from typing import Dict, List, Any
import requests


def generar_feedback_llm(
    metrica_dict: Dict[str, float],
    resumen_global: Dict[str, Any],
    eventos: List[str],
    puntos_a_mejorar: List[str],
    puntos_debiles: List[str],
    model: str = "llama3",
    timeout: int = 60,
) -> str:
    """Llama a un LLM local (Ollama) para generar feedback personalizado."""

    prompt = f"""
Eres un coach experto en oratoria y comunicación. Analiza los siguientes datos de una presentación grabada en video y genera un feedback personalizado, motivador y constructivo en español. Resalta los puntos fuertes, señala los aspectos a mejorar y da recomendaciones prácticas. Sé concreto y adapta el mensaje a los resultados.

Métricas globales:
- Fluidez: {metrica_dict['fluency']:.2f}
- Claridad: {metrica_dict['clarity']:.2f}
- Ritmo: {metrica_dict['pace']:.2f}
- Confianza: {metrica_dict['confidence']:.2f}
- Pronunciación: {metrica_dict['pronunciation']:.2f}
- Engagement: {metrica_dict['engagement']:.2f}

Resumen:
{resumen_global}

Eventos detectados:
{eventos}

Recomendaciones:
- Puntos a mejorar: {puntos_a_mejorar}
- Puntos débiles: {puntos_debiles}

Por favor, escribe el feedback en español, en tono positivo y claro.
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=timeout,
        )
        if response.ok:
            return response.json().get("response", "No se pudo obtener feedback del LLM.").strip()
        return (
            f"No se pudo conectar con el modelo LLM local (Ollama). "
            f"Error: {response.status_code} - {response.text}"
        )
    except Exception as exc:
        return f"Error al conectar con LLM local: {exc}"
