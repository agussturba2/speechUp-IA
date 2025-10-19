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
        logger.info(f"🔍 Worker polling Redis queue '{REDIS_CLIP_QUEUE}'...")
        job = await redis_conn.blpop(REDIS_CLIP_QUEUE, timeout=1)
        if not job:
            return

        _, job_id = job
        logger.info(f"📥 Received job from queue: {job_id}")
        job_key = f"{REDIS_CLIP_HASH_PREFIX}{job_id}"
        payload = await redis_conn.hgetall(job_key)
        if not payload:
            logger.warning(f"⚠️ Clip job {job_id} not found in Redis (key={job_key})")
            return

        clip_job = ClipJob.from_redis(payload)
        logger.info(f"🎬 Processing clip job {job_id}: type={clip_job.event_type}, start={clip_job.start_sec:.2f}s, end={clip_job.end_sec:.2f}s")

        if not clip_job.video_path:
            logger.error(f"❌ Clip job {job_id} missing video_path")
            return

        if not self.clip_bucket:
            logger.error(f"❌ CLIP_BUCKET not configured (current: {self.clip_bucket})")
            raise RuntimeError("CLIP_BUCKET environment variable is not set")
        if not self.clips_api_url:
            logger.error(f"❌ CLIPS_API_URL not configured (current: {self.clips_api_url})")
            raise RuntimeError("CLIPS_API_URL environment variable is not set")
        
        logger.info(f"✅ Configuration validated: bucket={self.clip_bucket}, api={self.clips_api_url}")

        clip_path: Optional[Path] = None
        thumbnail_path: Optional[Path] = None
        local_video_path: Optional[Path] = None

        try:
            # Download video from S3 if it's an S3 URL
            if clip_job.video_path and clip_job.video_path.startswith("s3://"):
                logger.info(f"📥 Downloading video from S3: {clip_job.video_path}")
                s3_client = await get_s3_client()
                
                # Parse S3 URL: s3://bucket/key
                s3_parts = clip_job.video_path.replace("s3://", "").split("/", 1)
                bucket = s3_parts[0]
                key = s3_parts[1] if len(s3_parts) > 1 else ""
                
                # Download to temp location
                local_video_path = Path(tempfile.gettempdir()) / f"{clip_job.session_id}_source.avi"
                await asyncio.to_thread(
                    s3_client.download_file,
                    bucket,
                    key,
                    str(local_video_path)
                )
                file_size = local_video_path.stat().st_size
                logger.info(f"✅ Video downloaded from S3: {file_size} bytes")
                
                # Update job to use local path for processing
                clip_job.video_path = str(local_video_path)
            else:
                # Validate local video file exists
                logger.info(f"🔍 Validating video file: {clip_job.video_path}")
                if not Path(clip_job.video_path).exists():
                    logger.error(f"❌ Video file not found: {clip_job.video_path}")
                    await redis_conn.delete(job_key)
                    return
                
                file_size = Path(clip_job.video_path).stat().st_size
                logger.info(f"✅ Video file validated: {file_size} bytes")

            # Generate clip and thumbnail in parallel
            logger.info(f"⚙️ Starting clip generation (parallel)...")
            clip_task = self._generate_clip(clip_job)
            thumbnail_task = self._generate_thumbnail_direct(clip_job)
            clip_path, thumbnail_path = await asyncio.gather(clip_task, thumbnail_task)
            logger.info(f"✅ Clip and thumbnail generated: clip={clip_path}, thumb={thumbnail_path}")

            logger.info(f"☁️ Uploading to S3 bucket '{self.clip_bucket}'...")
            clip_url, thumbnail_url = await self._upload_to_s3(clip_job, clip_path, thumbnail_path)
            logger.info(f"✅ S3 upload complete: clip={clip_url}, thumb={thumbnail_url}")
            
            logger.info(f"📡 Notifying backend at {self.clips_api_url}...")
            await self._notify_backend(clip_job, clip_url, thumbnail_url)
            logger.info(f"✅ Backend notified successfully")
            
            # Delete source video from S3 if it was uploaded temporarily
            if local_video_path:
                try:
                    s3_client = await get_s3_client()
                    s3_parts = clip_job.video_path.replace("s3://", "").split("/", 1) if "s3://" in str(clip_job.video_path) else None
                    if s3_parts and len(s3_parts) == 2:
                        bucket, key = s3_parts[0], s3_parts[1]
                        # Get original S3 path from job metadata
                        original_s3_path = None
                        job_data = await redis_conn.hgetall(job_key)
                        if "video_path" in job_data:
                            original_s3_path = job_data["video_path"]
                        
                        if original_s3_path and original_s3_path.startswith("s3://"):
                            s3_parts = original_s3_path.replace("s3://", "").split("/", 1)
                            bucket, key = s3_parts[0], s3_parts[1]
                            await asyncio.to_thread(s3_client.delete_object, Bucket=bucket, Key=key)
                            logger.info(f"🗑️ Deleted temp video from S3: {original_s3_path}")
                except Exception as cleanup_err:
                    logger.warning(f"⚠️ Failed to delete temp video from S3: {cleanup_err}")
            
            await redis_conn.delete(job_key)
            logger.info(f"✅ Clip job {job_id} completed successfully")
        except Exception as exc:
            logger.error(f"❌ Failed processing clip job {job_id}: {exc}", exc_info=True)
            
            # Retry logic with max attempts
            if clip_job.attempts >= self.max_retries:
                await redis_conn.rpush("clip_jobs:failed", job_id)
                logger.error(f"💀 Job {job_id} exceeded max retries ({self.max_retries}), moved to failed queue")
            else:
                new_attempts = clip_job.attempts + 1
                backoff_time = 2 ** clip_job.attempts
                logger.warning(f"🔄 Retrying job {job_id} (attempt {new_attempts}/{self.max_retries}) after {backoff_time}s backoff")
                await redis_conn.hincrby(job_key, "attempts", 1)
                # Exponential backoff
                await asyncio.sleep(backoff_time)
                await redis_conn.rpush(REDIS_CLIP_QUEUE, job_id)
                logger.info(f"🔄 Job {job_id} re-enqueued")
        finally:
            # Clean up local files
            if clip_path and clip_path.exists():
                clip_path.unlink(missing_ok=True)
            if thumbnail_path and thumbnail_path.exists():
                thumbnail_path.unlink(missing_ok=True)
            if local_video_path and local_video_path.exists():
                local_video_path.unlink(missing_ok=True)
                logger.info(f"🧹 Cleaned up local video file: {local_video_path}")

    async def _generate_clip(self, job: ClipJob) -> Path:
        start = max(job.start_sec - job.margin_sec, 0.0)
        duration = (job.end_sec + job.margin_sec) - start
        clip_path = Path(tempfile.gettempdir()) / f"{job.job_id}.mp4"
        
        logger.info(f"🎞️ Generating clip: start={start:.2f}s, duration={duration:.2f}s -> {clip_path}")

        process = ffmpeg.input(job.video_path, ss=start, t=duration)
        output = ffmpeg.output(
            process,
            str(clip_path),
            vcodec="libx264",
            preset="veryfast",
            acodec="aac",
            movflags="faststart",
        )
        try:
            await asyncio.to_thread(output.run, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            logger.error(f"❌ FFmpeg error generating clip:")
            logger.error(f"   Command: {' '.join(output.compile())}")
            logger.error(f"   Stdout: {e.stdout.decode() if e.stdout else 'N/A'}")
            logger.error(f"   Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
            raise
        logger.info(f"✅ Clip generated: {clip_path.stat().st_size} bytes")
        return clip_path

    async def _generate_thumbnail_direct(self, job: ClipJob) -> Path:
        """Generate thumbnail directly from source video (parallel with clip generation)."""
        midpoint = (job.start_sec + job.end_sec) / 2.0
        thumbnail_path = Path(tempfile.gettempdir()) / f"{job.job_id}.jpg"
        
        logger.info(f"📸 Generating thumbnail at {midpoint:.2f}s -> {thumbnail_path}")
        process = ffmpeg.input(job.video_path, ss=midpoint)
        output = ffmpeg.output(process, str(thumbnail_path), vframes=1)
        try:
            await asyncio.to_thread(output.run, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            logger.error(f"❌ FFmpeg error generating thumbnail:")
            logger.error(f"   Command: {' '.join(output.compile())}")
            logger.error(f"   Stdout: {e.stdout.decode() if e.stdout else 'N/A'}")
            logger.error(f"   Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
            raise
        logger.info(f"✅ Thumbnail generated: {thumbnail_path.stat().st_size} bytes")
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
    logger.info("🚀 Starting IncrementalClipWorker...")
    worker = IncrementalClipWorker()
    logger.info(f"⚙️ Worker configuration:")
    logger.info(f"   - Redis URL: {worker.redis_url}")
    logger.info(f"   - S3 Bucket: {worker.clip_bucket}")
    logger.info(f"   - API URL: {worker.clips_api_url}")
    logger.info(f"   - Max Retries: {worker.max_retries}")
    logger.info(f"🔄 Worker loop started, waiting for jobs...")
    try:
        while True:
            await worker.run_once()
    except KeyboardInterrupt:
        logger.info("⏸️ Worker stopped by user")
    finally:
        logger.info("🛑 Shutting down worker...")
        await worker.close()
        logger.info("✅ Worker shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
