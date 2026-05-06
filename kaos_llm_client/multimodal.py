"""Convenience helpers for building multimodal message content.

All functions return dicts in OpenAI content-part format (the canonical
input format for kaos-llm-client). Provider-specific conversion happens
automatically in each provider's ``_build_request``.

Usage::

    from kaos_llm_client.multimodal import image_from_path, audio_from_path

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                image_from_path("photo.jpg"),
            ],
        }
    ]
    response = client.chat(messages)
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any


def image_url(url: str, *, detail: str | None = None) -> dict[str, Any]:
    """Image from URL (HTTP or data URI).

    Returns an OpenAI ``image_url`` content part.

    Args:
        url: HTTP URL or ``data:image/...;base64,...`` data URI.
        detail: OpenAI detail level — ``"low"``, ``"high"``, or ``"auto"``.
    """
    image_url_obj: dict[str, Any] = {"url": url}
    if detail is not None:
        image_url_obj["detail"] = detail
    return {"type": "image_url", "image_url": image_url_obj}


def image_from_bytes(data: bytes, media_type: str = "image/png") -> dict[str, Any]:
    """Image from raw bytes. Returns a data URI ``image_url`` part."""
    b64 = base64.b64encode(data).decode("ascii")
    return image_url(f"data:{media_type};base64,{b64}")


def image_from_path(path: str | Path) -> dict[str, Any]:
    """Read an image file and return a data URI ``image_url`` part.

    Media type is inferred from the file extension.
    """
    p = Path(path)
    media_type, _ = mimetypes.guess_type(str(p))
    if media_type is None:
        media_type = "image/png"
    return image_from_bytes(p.read_bytes(), media_type)


def audio_input(data: bytes | str, *, format: str = "wav") -> dict[str, Any]:
    """Audio input for OpenAI models.

    Returns an OpenAI ``input_audio`` content part. Google converts this
    to ``inline_data`` automatically. Anthropic does not support audio input.

    Args:
        data: Raw audio bytes, or a base64-encoded string.
        format: Audio format — ``"wav"``, ``"mp3"``, etc.
    """
    b64 = base64.b64encode(data).decode("ascii") if isinstance(data, bytes) else data
    return {"type": "input_audio", "input_audio": {"data": b64, "format": format}}


def audio_from_path(path: str | Path) -> dict[str, Any]:
    """Read an audio file and return an ``input_audio`` part.

    Format is inferred from the file extension.
    """
    p = Path(path)
    ext = p.suffix.lower()
    format_map: dict[str, str] = {
        ".wav": "wav",
        ".mp3": "mp3",
        ".ogg": "ogg",
        ".flac": "flac",
        ".aac": "aac",
        ".aiff": "aiff",
    }
    fmt = format_map.get(ext, "wav")
    return audio_input(p.read_bytes(), format=fmt)


def document_url(url: str, *, media_type: str = "application/pdf") -> dict[str, Any]:
    """Document from URL.

    Returns a ``document`` content part with URL source. Anthropic supports
    this natively for PDFs. Google converts to ``file_data`` for GCS URIs.

    Args:
        url: HTTP URL or GCS URI (``gs://...``).
        media_type: MIME type of the document.
    """
    return {
        "type": "document",
        "source": {"type": "url", "url": url, "media_type": media_type},
    }


def document_from_bytes(data: bytes, media_type: str = "application/pdf") -> dict[str, Any]:
    """Document from raw bytes. Returns a base64 ``document`` content part."""
    b64 = base64.b64encode(data).decode("ascii")
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": media_type, "data": b64},
    }


def document_from_path(path: str | Path) -> dict[str, Any]:
    """Read a document file (PDF, etc.) and return a ``document`` content part.

    Media type is inferred from the file extension.
    """
    p = Path(path)
    media_type, _ = mimetypes.guess_type(str(p))
    if media_type is None:
        media_type = "application/pdf"
    return document_from_bytes(p.read_bytes(), media_type)
