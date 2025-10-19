"""Basic timeline event extraction for incremental analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class TimelineEvent:
    """Normalized representation of a highlight or issue detected in a session."""

    type: str
    start: float
    end: Optional[float] = None
    severity: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload: Dict[str, Any] = {
            "type": self.type,
            "start": round(self.start, 4),
        }
        if self.end is not None:
            payload["end"] = round(self.end, 4)
        if self.severity:
            payload["severity"] = self.severity
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class TimelineGenerator:
    """Utility to normalize events from analysis results."""

    def __init__(self, default_severity: str = "medium") -> None:
        self.default_severity = default_severity

    def events_from_result(self, result: Dict[str, Any]) -> List[TimelineEvent]:
        """Extract timeline events from the aggregated analysis result."""
        raw_events: Iterable[Dict[str, Any]] = result.get("events", []) or []
        events: List[TimelineEvent] = []

        for entry in raw_events:
            event_type = entry.get("type") or entry.get("kind")
            if not event_type:
                continue

            start = self._extract_start(entry)
            end = self._extract_end(entry, start)
            severity = entry.get("severity") or self.default_severity
            metadata = self._extract_metadata(entry)

            events.append(
                TimelineEvent(
                    type=event_type,
                    start=start,
                    end=end,
                    severity=severity,
                    metadata=metadata,
                )
            )

        return events

    def build_timeline(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Return a summarized timeline structure for downstream consumers."""
        events = [event.to_dict() for event in self.events_from_result(result)]
        return {
            "duration": result.get("media", {}).get("duration_sec"),
            "events": events,
        }

    @staticmethod
    def _extract_start(entry: Dict[str, Any]) -> float:
        if "start" in entry:
            return float(entry["start"])
        if "t" in entry:
            return float(entry["t"])
        if "time_sec" in entry:
            return float(entry["time_sec"])
        return float(entry.get("timestamp", 0.0))

    @staticmethod
    def _extract_end(entry: Dict[str, Any], default_start: float) -> Optional[float]:
        if "end" in entry:
            return float(entry["end"])
        duration = entry.get("duration")
        if duration is not None:
            try:
                return default_start + float(duration)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _extract_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
        metadata_keys = {"label", "score", "confidence", "word", "count"}
        metadata: Dict[str, Any] = {}
        for key in metadata_keys:
            if key in entry:
                metadata[key] = entry[key]
        return metadata
