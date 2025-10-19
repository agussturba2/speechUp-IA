"""Async helpers for uploading clips to S3."""

from __future__ import annotations

import asyncio
from typing import Optional

import boto3
from botocore.config import Config

_s3_client: Optional[boto3.client] = None
_lock = asyncio.Lock()


async def get_s3_client(region: Optional[str] = None) -> boto3.client:
    global _s3_client
    if _s3_client is None:
        async with _lock:
            if _s3_client is None:
                session = boto3.session.Session()
                _s3_client = session.client(
                    "s3",
                    region_name=region,
                    config=Config(signature_version="s3v4"),
                )
    return _s3_client
