"""Worker that processes clip jobs from Redis."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import ffmpeg
import redis.asyncio as redis
import httpx

from api.services.clip_scheduler import (
    ClipJob,
    REDIS_CLIP_HASH_PREFIX,
    REDIS_CLIP_QUEUE,
)
from api.services.db import get_database
from api.services.s3_client import get_s3_client

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_CLIP_BUCKET = "clips-bucket-speech-up"
DEFAULT_CLIPS_API_URL = "http://98.91.55.213:7070"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


class IncrementalClipWorker:
    """Consume clip jobs, generate clips, upload to S3, and persist metadata."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        clip_bucket: Optional[str] = None,
        clips_api_url: Optional[str] = None,
    ) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL") or DEFAULT_REDIS_URL
        self.clip_bucket = clip_bucket or os.getenv("CLIP_BUCKET") or DEFAULT_CLIP_BUCKET
        self.clips_api_url = clips_api_url or os.getenv("CLIPS_API_URL") or DEFAULT_CLIPS_API_URL
        self._redis: Optional[redis.Redis] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self.max_retries = 3

    async def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def run_once(self) -> None:
        """Process a single job from the queue if available."""
        redis_conn = await self._get_redis()
        job = await redis_conn.blpop(REDIS_CLIP_QUEUE, timeout=1)
        if not job:
            return

        _, job_id = job
        job_key = f"{REDIS_CLIP_HASH_PREFIX}{job_id}"
        payload = await redis_conn.hgetall(job_key)
        if not payload:
            logger.warning("Clip job %s not found", job_id)
            return

        clip_job = ClipJob.from_redis(payload)
        logger.info("Processing clip job %s (type=%s)", job_id, clip_job.event_type)

        if not clip_job.video_path:
            logger.warning("Clip job %s missing video_path", job_id)
            return

        if not self.clip_bucket:
            raise RuntimeError("CLIP_BUCKET environment variable is not set")
        if not self.clips_api_url:
            raise RuntimeError("CLIPS_API_URL environment variable is not set")

        clip_path: Optional[Path] = None
        thumbnail_path: Optional[Path] = None

        try:
            # Validate video file exists
            if not Path(clip_job.video_path).exists():
                logger.error("Video file not found: %s", clip_job.video_path)
                await redis_conn.delete(job_key)
                return

            # Generate clip and thumbnail in parallel
            clip_task = self._generate_clip(clip_job)
            thumbnail_task = self._generate_thumbnail_direct(clip_job)
            clip_path, thumbnail_path = await asyncio.gather(clip_task, thumbnail_task)

            clip_url, thumbnail_url = await self._upload_to_s3(clip_job, clip_path, thumbnail_path)
            await self._notify_backend(clip_job, clip_url, thumbnail_url)
            await redis_conn.delete(job_key)
            logger.info("Clip job %s completed", job_id)
        except Exception as exc:
            logger.error("Failed processing clip job %s: %s", job_id, exc, exc_info=True)
            
            # Retry logic with max attempts
            if clip_job.attempts >= self.max_retries:
                await redis_conn.rpush("clip_jobs:failed", job_id)
                logger.error("Job %s exceeded max retries (%d)", job_id, self.max_retries)
            else:
                await redis_conn.hincrby(job_key, "attempts", 1)
                # Exponential backoff
                await asyncio.sleep(2 ** clip_job.attempts)
                await redis_conn.rpush(REDIS_CLIP_QUEUE, job_id)
        finally:
            if clip_path and clip_path.exists():
                clip_path.unlink(missing_ok=True)
            if thumbnail_path and thumbnail_path.exists():
                thumbnail_path.unlink(missing_ok=True)

    async def _generate_clip(self, job: ClipJob) -> Path:
        start = max(job.start_sec - job.margin_sec, 0.0)
        duration = (job.end_sec + job.margin_sec) - start
        clip_path = Path(tempfile.gettempdir()) / f"{job.job_id}.mp4"

        process = ffmpeg.input(job.video_path, ss=start, t=duration)
        output = ffmpeg.output(
            process,
            str(clip_path),
            vcodec="libx264",
            acodec="aac",
            preset="veryfast",
            movflags="faststart",
        )
        await asyncio.to_thread(output.run, overwrite_output=True, quiet=True)
        return clip_path

    async def _generate_thumbnail_direct(self, job: ClipJob) -> Path:
        """Generate thumbnail directly from source video (parallel with clip generation)."""
        midpoint = (job.start_sec + job.end_sec) / 2.0
        thumbnail_path = Path(tempfile.gettempdir()) / f"{job.job_id}.jpg"
        process = ffmpeg.input(job.video_path, ss=midpoint)
        output = ffmpeg.output(process, str(thumbnail_path), vframes=1)
        await asyncio.to_thread(output.run, overwrite_output=True, quiet=True)
        return thumbnail_path

    async def _upload_to_s3(self, job: ClipJob, clip_path: Path, thumbnail_path: Path) -> Tuple[str, str]:
        s3_client = await get_s3_client()
        clip_key = f"session-{job.session_id}/{job.job_id}.mp4"
        thumbnail_key = f"session-{job.session_id}/{job.job_id}.jpg"

        # Upload both files in parallel
        await asyncio.gather(
            asyncio.to_thread(
                s3_client.upload_file,
                str(clip_path),
                self.clip_bucket,
                clip_key,
                ExtraArgs={"ContentType": "video/mp4"},
            ),
            asyncio.to_thread(
                s3_client.upload_file,
                str(thumbnail_path),
                self.clip_bucket,
                thumbnail_key,
                ExtraArgs={"ContentType": "image/jpeg"},
            )
        )

        clip_url = f"s3://{self.clip_bucket}/{clip_key}"
        thumbnail_url = f"s3://{self.clip_bucket}/{thumbnail_key}"
        return clip_url, thumbnail_url

    async def _notify_backend(self, job: ClipJob, clip_url: str, thumbnail_url: str) -> None:
        duration = job.duration_sec or max(job.end_sec - job.start_sec, 0.0)
        payload = {
            "sessionId": job.session_id,
            "userId": job.session_id,  # adjust if userId is tracked separately
            "eventType": job.event_type,
            "startSec": f"{job.start_sec:.4f}",
            "endSec": f"{job.end_sec:.4f}",
            "durationSec": f"{duration:.4f}",
            "clipUrl": clip_url,
            "thumbnailUrl": thumbnail_url,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }

        # Use persistent HTTP client with connection pooling
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        
        response = await self._http_client.post(
            self.clips_api_url.rstrip("/") + "/clips",
            json=payload,
        )
        response.raise_for_status()

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


async def main() -> None:
    worker = IncrementalClipWorker()
    try:
        while True:
            await worker.run_once()
    finally:
        await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
