"""Schedule clip generation jobs in Redis."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)

REDIS_CLIP_QUEUE = "clip_jobs:pending"
REDIS_CLIP_HASH_PREFIX = "clip_job:"

TARGET_EVENTS = {"filler", "pause_long", "good_rhythm", "gaze_lost"}
DEFAULT_MARGIN_SEC = 0.5
MAX_CLIPS_PER_SESSION = 10
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


@dataclass
class ClipJob:
    """Represent a clip generation job stored in Redis."""

    job_id: str
    session_id: str
    event_type: str
    start_sec: float
    end_sec: float
    margin_sec: float = DEFAULT_MARGIN_SEC
    attempts: int = 0
    status: str = "pending"
    video_path: Optional[str] = None
    duration_sec: Optional[float] = None

    def to_redis(self) -> dict[str, str]:
        payload = asdict(self)
        payload["job_id"] = self.job_id
        payload["start_sec"] = f"{self.start_sec:.4f}"
        payload["end_sec"] = f"{self.end_sec:.4f}"
        payload["margin_sec"] = f"{self.margin_sec:.4f}"
        payload["attempts"] = str(self.attempts)
        payload["status"] = self.status
        if self.video_path is None:
            payload.pop("video_path", None)
        if self.duration_sec is None:
            payload.pop("duration_sec", None)
        else:
            payload["duration_sec"] = f"{self.duration_sec:.4f}"
        return payload

    @classmethod
    def from_redis(cls, data: dict[str, str]) -> "ClipJob":
        """Create a job from Redis hash data."""

        def _parse_float(value: Optional[str], default: float = 0.0) -> float:
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        return cls(
            job_id=data["job_id"],
            session_id=data.get("session_id", "unknown"),
            event_type=data.get("event_type", "unknown"),
            start_sec=_parse_float(data.get("start_sec")),
            end_sec=_parse_float(data.get("end_sec")),
            margin_sec=_parse_float(data.get("margin_sec"), DEFAULT_MARGIN_SEC),
            attempts=int(data.get("attempts", "0")),
            status=data.get("status", "pending"),
            video_path=data.get("video_path"),
            duration_sec=float(data["duration_sec"]) if data.get("duration_sec") else None,
        )


class ClipScheduler:
    """Enqueue clip generation jobs in Redis."""

    def __init__(self, redis_url: Optional[str] = None) -> None:
        """Create a scheduler backed by Redis."""

        self._redis_url = redis_url or os.getenv("REDIS_URL") or DEFAULT_REDIS_URL
        self._redis: Optional[redis.Redis] = None

    async def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def enqueue_session(
        self,
        session_id: str,
        events: Iterable[dict],
        *,
        video_path: Optional[str] = None,
        duration_sec: Optional[float] = None,
    ) -> int:
        """Push clip jobs for the given session events."""

        redis = await self._get_redis()
        enqueued = 0
        jobs = []

        for index, event in enumerate(events):
            if index >= MAX_CLIPS_PER_SESSION:
                logger.info("Reached clip limit for session %s", session_id)
                break

            event_type = event.get("type")
            if event_type not in TARGET_EVENTS:
                continue

            start_sec = float(event.get("start", 0.0))
            end_sec = float(event.get("end", start_sec))

            job = ClipJob(
                job_id=str(uuid.uuid4()),
                session_id=session_id,
                event_type=event_type,
                start_sec=start_sec,
                end_sec=end_sec,
                video_path=video_path,
                duration_sec=duration_sec,
            )
            jobs.append(job)

        # Use Redis pipeline for batch operations
        if jobs:
            async with redis.pipeline(transaction=True) as pipe:
                for job in jobs:
                    pipe.hset(f"{REDIS_CLIP_HASH_PREFIX}{job.job_id}", mapping=job.to_redis())
                    pipe.rpush(REDIS_CLIP_QUEUE, job.job_id)
                await pipe.execute()
                enqueued = len(jobs)
                logger.info("Enqueued %d clip jobs for session %s", enqueued, session_id)

        return enqueued

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
            self._redis = None


__all__ = [
    "ClipScheduler",
    "ClipJob",
    "TARGET_EVENTS",
    "REDIS_CLIP_QUEUE",
    "REDIS_CLIP_HASH_PREFIX",
]
