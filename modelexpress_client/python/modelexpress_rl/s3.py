# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct immutable S3 writes for canonical refit artifacts."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, wait
from urllib.parse import urlsplit

from modelexpress_rl import envs as rl_envs


_MIN_MULTIPART_PART_BYTES = 5 * 1024**2
_MAX_MULTIPART_PART_BYTES = 5 * 1024**3
_MAX_MULTIPART_PARTS = 10_000
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
logger = logging.getLogger(__name__)


class ImmutableS3Conflict(RuntimeError):
    """An immutable key already contains different bytes."""


def _error_code(error: Exception) -> str | None:
    try:
        return str(error.response["Error"]["Code"])  # type: ignore[attr-defined]
    except (AttributeError, KeyError, TypeError):
        return None


def _parse_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.startswith("/")
        or parsed.path.startswith("//")
        or len(parsed.path) == 1
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"invalid S3 URI: {uri!r}")
    return parsed.netloc, parsed.path[1:]


def _read_body(body) -> bytes:
    try:
        return body.read()
    finally:
        body.close()


class S3Client:
    """Small immutable S3 client for canonical refit artifacts."""

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ) -> None:
        import boto3
        from botocore.config import Config as BotoConfig

        self._multipart_threshold_bytes = rl_envs.MX_S3_MULTIPART_THRESHOLD_BYTES
        self._upload_part_bytes = rl_envs.MX_S3_UPLOAD_PART_BYTES
        if not (
            _MIN_MULTIPART_PART_BYTES
            <= self._upload_part_bytes
            <= _MAX_MULTIPART_PART_BYTES
        ):
            raise ValueError("MX_S3_UPLOAD_PART_BYTES must be between 5 MiB and 5 GiB")
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            config=BotoConfig(
                max_pool_connections=rl_envs.MX_S3_MAX_POOL_CONNECTIONS,
                retries={
                    "total_max_attempts": rl_envs.MX_S3_MAX_ATTEMPTS,
                    "mode": "standard",
                },
                tcp_keepalive=rl_envs.MX_S3_TCP_KEEPALIVE,
            ),
        )
        self._upload_pool = ThreadPoolExecutor(
            max_workers=rl_envs.MX_S3_UPLOAD_WORKERS,
            thread_name_prefix="modelexpress-s3-upload",
        )
        self._range_bytes = rl_envs.MX_S3_DOWNLOAD_RANGE_BYTES
        self._range_threshold_bytes = rl_envs.MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES
        self._max_attempts = rl_envs.MX_S3_MAX_ATTEMPTS
        # Range parts never submit work, so callers on other pools cannot deadlock.
        self._download_pool = ThreadPoolExecutor(
            max_workers=rl_envs.MX_S3_DOWNLOAD_WORKERS,
            thread_name_prefix="modelexpress-s3-range",
        )

    def put(self, *, uri: str, data: bytes) -> None:
        """Create an immutable object, accepting an identical retry."""
        bucket, key = _parse_uri(uri)
        if len(data) >= self._multipart_threshold_bytes:
            self._put_multipart(bucket=bucket, key=key, uri=uri, data=data)
            return
        try:
            self._client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                IfNoneMatch="*",
            )
        except Exception as error:
            if _error_code(error) not in {"412", "PreconditionFailed"}:
                raise
            existing = self.get(uri)
            if existing != data:
                raise ImmutableS3Conflict(
                    f"immutable S3 object conflict for {bucket}/{key}"
                ) from error

    def _put_multipart(
        self,
        *,
        bucket: str,
        key: str,
        uri: str,
        data: bytes,
    ) -> None:
        part_count = (
            len(data) + self._upload_part_bytes - 1
        ) // self._upload_part_bytes
        if part_count > _MAX_MULTIPART_PARTS:
            raise ValueError("multipart upload exceeds 10,000 parts")
        upload_id = self._client.create_multipart_upload(
            Bucket=bucket,
            Key=key,
        )["UploadId"]

        def upload_part(index: int) -> dict[str, int | str]:
            start = index * self._upload_part_bytes
            response = self._client.upload_part(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=index + 1,
                Body=data[start : start + self._upload_part_bytes],
            )
            return {"ETag": response["ETag"], "PartNumber": index + 1}

        try:
            futures = [
                self._upload_pool.submit(upload_part, index)
                for index in range(part_count)
            ]
            try:
                parts = [future.result() for future in futures]
            except Exception:
                wait(futures)
                raise
            self._client.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
                IfNoneMatch="*",
            )
        except Exception as error:
            try:
                self._client.abort_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                )
            except Exception:
                logger.warning("Failed to abort multipart upload", exc_info=True)
            if _error_code(error) not in {"412", "PreconditionFailed"}:
                raise
            existing = self.get(uri)
            if existing != data:
                raise ImmutableS3Conflict(
                    f"immutable S3 object conflict for {bucket}/{key}"
                ) from error

    def get(self, uri: str) -> bytes:
        """Read one S3 object into a preallocated buffer with ranged GETs.

        The first range reports the object size and ETag; remaining ranges are
        pinned to that ETag and fetched in parallel when the object is large.
        """
        bucket, key = _parse_uri(uri)
        try:
            response = self._client.get_object(
                Bucket=bucket, Key=key, Range=f"bytes=0-{self._range_bytes - 1}"
            )
        except Exception as error:
            if _error_code(error) not in {"416", "InvalidRange"}:
                raise
            # An empty object has no satisfiable range.
            return _read_body(self._client.get_object(Bucket=bucket, Key=key)["Body"])
        head = _read_body(response["Body"])
        match = _CONTENT_RANGE.match(response.get("ContentRange") or "")
        if match is None:
            # The server ignored the range and returned the whole object.
            return head
        total = int(match.group(3))
        if len(head) != int(match.group(2)) - int(match.group(1)) + 1:
            raise RuntimeError(f"short S3 range read for {bucket}/{key}")
        if total == len(head):
            return head
        etag = response["ETag"]
        buffer = bytearray(total)
        view = memoryview(buffer)
        view[: len(head)] = head
        step = self._range_bytes if total >= self._range_threshold_bytes else total
        ranges = [
            (start, min(start + step, total)) for start in range(len(head), total, step)
        ]

        def fetch(span: tuple[int, int]) -> None:
            start, end = span
            for attempt in range(self._max_attempts):
                try:
                    part = self._client.get_object(
                        Bucket=bucket,
                        Key=key,
                        Range=f"bytes={start}-{end - 1}",
                        IfMatch=etag,
                    )
                    data = _read_body(part["Body"])
                    if len(data) != end - start:
                        raise RuntimeError(f"short S3 range read for {bucket}/{key}")
                except Exception as error:
                    if (
                        _error_code(error) in {"412", "PreconditionFailed"}
                        or attempt == self._max_attempts - 1
                    ):
                        raise
                    continue
                view[start:end] = data
                return

        futures = [self._download_pool.submit(fetch, span) for span in ranges]
        try:
            for future in futures:
                future.result()
        except Exception:
            wait(futures)
            raise
        return bytes(buffer)

    def size(self, uri: str) -> int:
        """Return one S3 object's byte size without downloading its payload."""
        bucket, key = _parse_uri(uri)
        return int(self._client.head_object(Bucket=bucket, Key=key)["ContentLength"])

    def close(self) -> None:
        """Close the underlying SDK client when supported."""
        self._upload_pool.shutdown()
        self._download_pool.shutdown()
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


__all__ = ["ImmutableS3Conflict", "S3Client"]
