from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator

VERSION = "1.0.0"

class Media(BaseModel):
    duration_sec: float
    lang: str = Field(default="es-AR")
    fps: int

class Scores(BaseModel):
    fluency: int = Field(..., ge=0, le=100)
    clarity: int = Field(..., ge=0, le=100)
    delivery_confidence: int = Field(..., ge=0, le=100)
    pronunciation: int = Field(..., ge=0, le=100)
    pace: int = Field(..., ge=0, le=100)
    engagement: int = Field(..., ge=0, le=100)

class Verbal(BaseModel):
    wpm: float
    articulation_rate_sps: float
    fillers_per_min: float
    filler_counts: Dict[str, int]
    avg_pause_sec: float
    pause_rate_per_min: float
    long_pauses: List[Dict[str, float]]
    pronunciation_score: float = Field(..., ge=0, le=1)
    stt_confidence: float = Field(..., ge=0, le=1)
    transcript_short: Optional[str] = None
    transcript_len: Optional[int] = None
    transcript_full: Optional[str] = Field(None, description="Full transcript (gated by SPEECHUP_INCLUDE_TRANSCRIPT)")

class Prosody(BaseModel):
    pitch_mean_hz: float
    pitch_range_semitones: float
    pitch_cv: float = Field(..., ge=0, le=1)
    energy_cv: float = Field(..., ge=0, le=1)
    rhythm_consistency: float = Field(..., ge=0, le=1)

class Lexical(BaseModel):
    lexical_diversity: float = Field(..., ge=0, le=1)
    cohesion_score: float = Field(..., ge=0, le=1)
    summary: str
    keywords: List[str]

class Nonverbal(BaseModel):
    gaze_screen_pct: float = Field(..., ge=0, le=1)
    head_stability: float = Field(..., ge=0, le=1)
    posture_openness: float = Field(..., ge=0, le=1)
    gesture_rate_per_min: float
    gesture_amplitude: float = Field(..., ge=0, le=1)
    expression_variability: float = Field(..., ge=0, le=1)
    face_coverage_pct: float = Field(..., ge=0, le=1)
    engagement: float = Field(..., ge=0, le=1)

class Event(BaseModel):
    t: float
    kind: str
    label: Optional[str] = None
    duration: Optional[float] = None
    amplitude: Optional[float] = Field(None, description="Gesture motion amplitude")
    score: Optional[float] = Field(None, description="Event confidence score")
    confidence: Optional[float] = Field(None, description="Event detection confidence")
    end_t: Optional[float] = Field(None, description="Event end timestamp")
    frame: Optional[int] = Field(None, description="Frame index where event was detected")

class Recommendation(BaseModel):
    area: str
    tip: str

class Quality(BaseModel):
    frames_analyzed: int = 0
    dropped_frames_pct: float = 0.0
    audio_snr_db: float | None = None
    analysis_ms: int = Field(0, ge=0)
    audio_available: bool = False
    debug: Optional[Dict[str, Any]] = Field(None, description="Debug information including gesture stats and ASR errors")         

class AnalysisJSON(BaseModel):
    id: str
    version: str = Field(default=VERSION)
    media: Media
    scores: Scores
    verbal: Verbal
    prosody: Prosody
    lexical: Lexical
    nonverbal: Nonverbal
    events: List[Event]
    recommendations: List[Recommendation]
    labels: List[str] = Field(default_factory=list, description="Human-friendly labels describing the analysis results")
    quality: Quality

    @classmethod
    def model_json_schema(cls) -> dict:
        return cls.model_json_schema()

    @field_validator('media', mode='before')
    def set_default_lang(cls, v):
        if isinstance(v, dict) and 'lang' not in v:
            v['lang'] = 'es-AR'
        return v
