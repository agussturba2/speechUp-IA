# /video/advice_generator.py

"""
Generates qualitative advice based on quantitative analysis results.
"""
from typing import List, Dict, Tuple
from video.video_config import ADVICE_THRESHOLDS

# Labeling thresholds for human-friendly insights
IDEAL_WPM = (120, 160)          # rango recomendado ES
WPM_SLOW = 110                   # <110 lento
WPM_FAST = 170                   # >170 rápido

FILLERS_LOW = 3                  # por minuto
FILLERS_HIGH = 8

# Pausas (por minuto y duración promedio en seg)
PAUSE_RATE_REGULAR_MAX = 30      # <=30 /min → "ritmo regular"
PAUSE_RATE_CHAOTIC_MIN = 40      # >=40 /min → "ritmo caótico"
AVG_PAUSE_LONG = 0.8             # >0.8s promedio → pausas largas

# Prosodia (coef. de variación / rango en semitonos)
PITCH_RANGE_FLAT = 4.0           # <4 st → "entonación plana"
PITCH_RANGE_EXPRESSIVE = 8.0     # ≥8 st → "entonación expresiva"
ENERGY_CV_LOW = 0.12             # bajo contraste de energía
ENERGY_CV_HIGH = 0.22            # buen contraste

# Gestos (por minuto)
GESTURE_RATE_LOW = 3.0
GESTURE_RATE_GOOD = 6.0


class AdviceGenerator:
    """Encapsulates logic for creating feedback from analysis metrics."""

    def __init__(self, thresholds: dict = ADVICE_THRESHOLDS):
        self.thresholds = thresholds

    def generate_labels_and_tips(self, proc: Dict) -> Tuple[List[str], List[Dict]]:
        """
        Generate human-friendly labels and recommendations based on analysis results.
        
        Args:
            proc: Dictionary containing verbal, prosody, and nonverbal metrics
            
        Returns:
            Tuple of (labels, recommendations) where labels is a list of strings
            and recommendations is a list of dicts with {area, tip}
        """
        labels = []
        recommendations = []
        
        verbal = proc.get("verbal", {})
        prosody = proc.get("prosody", {})
        nonverbal = proc.get("nonverbal", {})
        
        # Generate labels and tips for each area
        wpm_labels, wpm_tips = self._analyze_wpm(verbal)
        filler_labels, filler_tips = self._analyze_fillers(verbal)
        pause_labels, pause_tips = self._analyze_pauses(verbal)
        prosody_labels, prosody_tips = self._analyze_prosody(prosody)
        gesture_labels, gesture_tips = self._analyze_gestures(nonverbal)
        
        # Combine all labels (avoiding contradictions)
        labels.extend(wpm_labels)
        labels.extend(filler_labels)
        labels.extend(pause_labels)
        labels.extend(prosody_labels)
        labels.extend(gesture_labels)
        
        # Combine all recommendations
        recommendations.extend(wpm_tips)
        recommendations.extend(filler_tips)
        recommendations.extend(pause_tips)
        recommendations.extend(prosody_tips)
        recommendations.extend(gesture_tips)
        
        # Limit labels to maximum 5 and recommendations to maximum 6
        labels = labels[:5]
        recommendations = recommendations[:6]
        
        # Apply deduplication and balancing
        recommendations = self._balance_and_dedupe(recommendations, labels)
        
        return labels, recommendations

    def _analyze_wpm(self, verbal: Dict) -> Tuple[List[str], List[Dict]]:
        """Analyze speaking rate (WPM) and generate labels and tips."""
        labels = []
        tips = []
        
        wpm = verbal.get("wpm", 0.0)
        if wpm <= 0:
            return labels, tips
            
        if wpm < WPM_SLOW:
            labels.append("ritmo lento")
            tips.append({
                "area": "comunicación",
                "tip": "Tu ritmo es tranquilo. Probá subir un poco la velocidad en ideas secundarias para sostener la atención."
            })
        elif WPM_SLOW <= wpm <= WPM_FAST:
            labels.append("ritmo adecuado")
            tips.append({
                "area": "comunicación",
                "tip": "Buen ritmo: claro y fácil de seguir."
            })
        else:  # wpm > WPM_FAST
            labels.append("ritmo acelerado")
            tips.append({
                "area": "comunicación",
                "tip": "Vas rápido. Probá marcar micro-pausas al final de cada frase para respirar y enfatizar."
            })
            
        return labels, tips

    def _analyze_fillers(self, verbal: Dict) -> Tuple[List[str], List[Dict]]:
        """Analyze filler words and generate labels and tips."""
        labels = []
        tips = []
        
        fillers_per_min = verbal.get("fillers_per_min", 0.0)
        if fillers_per_min <= 0:
            return labels, tips
            
        if fillers_per_min <= FILLERS_LOW:
            labels.append("pocas muletillas")
            tips.append({
                "area": "comunicación",
                "tip": "¡Casi sin muletillas! Muy prolijo."
            })
        elif fillers_per_min >= FILLERS_HIGH:
            labels.append("muchas muletillas")
            tips.append({
                "area": "comunicación",
                "tip": "Hay varias muletillas. Ensayá 10s de silencio consciente: cuando aparezca 'este/eh', frená, respirá y seguí."
            })
            
        return labels, tips

    def _analyze_pauses(self, verbal: Dict) -> Tuple[List[str], List[Dict]]:
        """Analyze pause patterns and generate labels and tips."""
        labels = []
        tips = []
        
        pause_rate = verbal.get("pause_rate_per_min", 0.0)
        avg_pause = verbal.get("avg_pause_sec", 0.0)
        
        if pause_rate <= 0:
            return labels, tips
            
        if pause_rate <= PAUSE_RATE_REGULAR_MAX:
            labels.append("pausas regulares")
            tips.append({
                "area": "comunicación",
                "tip": "Buen uso de pausas: ayuda a procesar el mensaje."
            })
        elif pause_rate >= PAUSE_RATE_CHAOTIC_MIN:
            labels.append("pausas caóticas")
            tips.append({
                "area": "comunicación",
                "tip": "Pausas irregulares. Probá cerrar cada idea con una pausa breve (~0.5s) y luego continuar."
            })
            
        if avg_pause > AVG_PAUSE_LONG:
            labels.append("pausas largas")
            tips.append({
                "area": "comunicación",
                "tip": "Las pausas son largas. Reducí levemente la duración para no perder momentum."
            })
            
        return labels, tips

    def _analyze_prosody(self, prosody: Dict) -> Tuple[List[str], List[Dict]]:
        """Analyze prosody metrics and generate labels and tips."""
        labels = []
        tips = []
        
        pitch_range = prosody.get("pitch_range_semitones", 0.0)
        energy_cv = prosody.get("energy_cv", 0.0)
        
        if pitch_range <= 0:
            return labels, tips
            
        if pitch_range < PITCH_RANGE_FLAT:
            labels.append("entonación plana")
            tips.append({
                "area": "voz",
                "tip": "Variá la entonación en palabras clave (subí 2–3 semitonos en conceptos importantes)."
            })
        elif pitch_range >= PITCH_RANGE_EXPRESSIVE:
            labels.append("entonación expresiva")
            tips.append({
                "area": "voz",
                "tip": "Buena variación tonal: mantiene el interés."
            })
            
        if energy_cv < ENERGY_CV_LOW:
            labels.append("bajo contraste de energía")
            tips.append({
                "area": "voz",
                "tip": "Sumá contrastes de energía: enfatizá verbos/números con leve aumento de volumen."
            })
        elif energy_cv >= ENERGY_CV_HIGH:
            labels.append("buen contraste de energía")
            tips.append({
                "area": "voz",
                "tip": "Buen contraste de energía: hace más dinámico el discurso."
            })
            
        return labels, tips

    def _analyze_gestures(self, nonverbal: Dict) -> Tuple[List[str], List[Dict]]:
        """Analyze gesture patterns and generate labels and tips."""
        labels = []
        tips = []
        
        gesture_rate = nonverbal.get("gesture_rate_per_min", 0.0)
        if gesture_rate <= 0:
            return labels, tips
            
        if gesture_rate < GESTURE_RATE_LOW:
            labels.append("pocos gestos")
            tips.append({
                "area": "presencia",
                "tip": "Acompañá las ideas con gestos abiertos de manos a altura del pecho."
            })
        elif gesture_rate >= GESTURE_RATE_GOOD:
            labels.append("gestualidad activa")
            tips.append({
                "area": "presencia",
                "tip": "Buena gestualidad: refuerza el mensaje."
            })
            
        return labels, tips

    def generate_nonverbal_tips(self, nonverbal: dict) -> list:
        tips = []
        
        # Check specific nonverbal metrics and add tips
        if nonverbal.get('gaze_screen_pct', 1) < 0.8:
            tips.append("Mirá a cámara más seguido para sostener la conexión.")
        if nonverbal.get('gesture_amplitude', 1) < 0.3:
            tips.append("Aumentá un poco la amplitud de los gestos para enfatizar ideas.")
        if nonverbal.get('expression_variability', 1) < 0.5:
            tips.append("Variá más tus expresiones faciales para mantener el interés.")
        if nonverbal.get('posture_openness', 1) < 0.7:
            tips.append("Abrí hombros y mantené una postura más expansiva.")
        
        # Fallback: if no tips were added, append default tip
        if not tips:
            tips.append("Excelente base. Podés marcar contrastes de energía en los puntos clave.")
        
        return tips[:3]  # Keep max 3 tips

    def generate_verbal_tips(self, verbal: dict) -> list:
        """Generate verbal communication tips based on ASR analysis."""
        tips = []
        
        # WPM recommendations
        wpm = verbal.get('wpm', 0.0)
        if wpm > 0 and wpm < 80 and verbal.get('avg_pause_sec', 0) > 0:
            # Only suggest if we have actual speech data
            tips.append("Podés aumentar un poco el ritmo para mantener la atención.")
        
        # Filler word recommendations
        fillers_per_min = verbal.get('fillers_per_min', 0.0)
        if fillers_per_min > 10:
            tips.append("Reducí muletillas como 'este', 'o sea', 'eh' para sonar más claro.")
        
        return tips

    def generate_prosody_tips(self, prosody: dict) -> list:
        """Generate prosody-based communication tips."""
        tips = []
        
        # Pitch variation suggestions
        pitch_range = prosody.get('pitch_range_semitones', 0.0)
        if pitch_range < 2.0:
            tips.append("Variá más el tono de voz para enfatizar puntos clave.")
        
        # Energy dynamics suggestions
        energy_cv = prosody.get('energy_cv', 0.0)
        if energy_cv < 0.1:
            tips.append("Añadí variación en el volumen para crear más dinamismo.")
        
        # Rhythm consistency suggestions
        rhythm_consistency = prosody.get('rhythm_consistency', 0.0)
        if rhythm_consistency < 0.4:
            tips.append("Practicá el ritmo de las frases para mejorar la fluidez.")
        
        return tips[:2]  # Keep max 2 prosody tips

    def generate_global_advice(
            self, total_faces: int, total_posture: int, total_frames: int
    ) -> str:
        if total_frames == 0:
            return "No hay datos suficientes para generar recomendaciones."
        face_ratio = total_faces / total_frames
        posture_ratio = total_posture / total_frames if total_frames > 0 else 0
        advice_parts = [
            self._get_face_advice(face_ratio),
            self._get_posture_advice(posture_ratio)
        ]
        if (face_ratio < self.thresholds["face"]["good"] and
                posture_ratio < self.thresholds["posture"]["good"]):
            advice_parts.append(
                "Practica frente a un espejo para mejorar presencia y postura visual."
            )
        return " ".join(advice_parts)

    def _norm_key(self, txt: str) -> str:
        """Normalize text for deduplication comparison."""
        return " ".join(txt.lower().split())

    def _balance_and_dedupe(self, advice: List[Dict], labels: List[str]) -> List[Dict]:
        """
        Deduplicate advice and balance across areas to ensure variety.
        
        Args:
            advice: List of advice dictionaries with {area, tip}
            labels: List of current labels for context-aware filtering
            
        Returns:
            Balanced and deduplicated list of advice
        """
        import os
        
        # Definir want al inicio del método para asegurar que siempre esté disponible
        try:
            want = int(os.getenv("SPEECHUP_ADVICE_TARGET_COUNT", "5"))
        except (ValueError, TypeError):
            # En caso de error al convertir a entero, usar un valor predeterminado
            want = 5
        
        # Si no hay consejos, devolver una lista vacía
        if not advice:
            return []
        
        seen = set()
        out = []
        by_area = {"voz": 0, "comunicación": 0, "presencia": 0, "communication": 0}
        
        # Check if energy-related labels are already present
        energy_labels = {"bajo contraste de energía", "entonación plana"}
        energy_already = any(l in labels for l in energy_labels)
        energy_tip_kept = False
        
        for a in advice:
            # Create deduplication key
            key = f'{a.get("area", "")}|{self._norm_key(a.get("tip", ""))}'
            if key in seen:  # exact duplicate
                continue
                
            # Allow at most 2 tips per area
            area = a.get("area", "")
            if by_area.get(area, 0) >= 2:
                continue
                
            # If energy label is present, keep only ONE energy-related tip
            tip_txt = a.get("tip", "").lower()
            is_energy_tip = ("energ" in tip_txt) or ("tono" in tip_txt) or ("entonación" in tip_txt)
            if energy_already and is_energy_tip:
                if energy_tip_kept:
                    continue
                energy_tip_kept = True
            
            # Add to output
            seen.add(key)
            by_area[area] = by_area.get(area, 0) + 1
            out.append(a)
        
        # If we ended with too few areas represented, back-fill with variety
        if len(out) < want:
            fillers = [
                {"area": "comunicación", "tip": "Usá ejemplos concretos para que la idea quede más clara."},
                {"area": "presencia", "tip": "Sostené los gestos a lo largo de todo el mensaje, no solo al inicio."},
                {"area": "voz", "tip": "Marcá palabras clave con una leve pausa antes o después."},
            ]
            for f in fillers:
                if len(out) >= want:
                    break
                k = f'{f["area"]}|{self._norm_key(f["tip"])}'
                # Check if we can add this filler without exceeding area limits
                if k not in seen and by_area.get(f["area"], 0) < 2:
                    out.append(f)
                    seen.add(k)
                    by_area[f["area"]] = by_area.get(f["area"], 0) + 1
        
        return out[:want]
