"""Tests for multimodal support — BinaryData, content helpers, and response properties."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from kaos_llm_client.multimodal import (
    audio_from_path,
    audio_input,
    document_from_bytes,
    document_from_path,
    document_url,
    image_from_bytes,
    image_from_path,
    image_url,
)
from kaos_llm_client.types import (
    BinaryData,
    ContentPart,
    ProviderResponse,
)

# Minimal 8-byte PNG header (not a full valid PNG, but enough for file I/O tests)
PNG_HEADER = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# TestBinaryData
# ---------------------------------------------------------------------------


class TestBinaryData:
    def test_construction(self):
        bd = BinaryData(data="aGVsbG8=", media_type="image/png")
        assert bd.data == "aGVsbG8="
        assert bd.media_type == "image/png"

    def test_data_uri_property(self):
        bd = BinaryData(data="aGVsbG8=", media_type="image/png")
        assert bd.data_uri == "data:image/png;base64,aGVsbG8="

    def test_is_image(self):
        assert BinaryData(data="x", media_type="image/png").is_image is True
        assert BinaryData(data="x", media_type="audio/wav").is_image is False

    def test_is_audio(self):
        assert BinaryData(data="x", media_type="audio/wav").is_audio is True
        assert BinaryData(data="x", media_type="image/png").is_audio is False

    def test_is_document(self):
        assert BinaryData(data="x", media_type="application/pdf").is_document is True
        assert BinaryData(data="x", media_type="image/png").is_document is False

    def test_from_data_uri(self):
        original = BinaryData(data="aGVsbG8=", media_type="image/png")
        uri = original.data_uri
        parsed = BinaryData.from_data_uri(uri)
        assert parsed.data == original.data
        assert parsed.media_type == original.media_type

    def test_from_bytes(self):
        bd = BinaryData.from_bytes(b"hello", media_type="application/octet-stream")
        assert bd.data == base64.b64encode(b"hello").decode("ascii")
        assert bd.media_type == "application/octet-stream"

    def test_from_path(self, tmp_path: Path):
        png_file = tmp_path / "test.png"
        png_file.write_bytes(PNG_HEADER)
        bd = BinaryData.from_path(png_file)
        assert bd.media_type == "image/png"
        assert bd.to_bytes() == PNG_HEADER

    def test_to_bytes(self):
        bd = BinaryData.from_bytes(b"hello", media_type="text/plain")
        assert bd.to_bytes() == b"hello"


# ---------------------------------------------------------------------------
# TestImageHelpers
# ---------------------------------------------------------------------------


class TestImageHelpers:
    def test_image_url_basic(self):
        result = image_url("https://example.com/img.png")
        assert result == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"},
        }

    def test_image_url_with_detail(self):
        result = image_url("https://example.com/img.png", detail="high")
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "https://example.com/img.png"
        assert result["image_url"]["detail"] == "high"

    def test_image_from_bytes(self):
        raw = b"\x89PNG"
        result = image_from_bytes(raw)
        assert result["type"] == "image_url"
        url = result["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        # Verify round-trip: decode the base64 portion
        b64_part = url.split(",", 1)[1]
        assert base64.b64decode(b64_part) == raw

    def test_image_from_path(self, tmp_path: Path):
        jpg_file = tmp_path / "photo.jpg"
        jpg_file.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG SOI marker
        result = image_from_path(jpg_file)
        assert result["type"] == "image_url"
        url = result["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# TestAudioHelpers
# ---------------------------------------------------------------------------


class TestAudioHelpers:
    def test_audio_input_from_bytes(self):
        raw = b"\x00\x01\x02\x03"
        result = audio_input(raw, format="wav")
        assert result == {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(raw).decode("ascii"),
                "format": "wav",
            },
        }

    def test_audio_input_from_base64_string(self):
        b64_str = base64.b64encode(b"audio-data").decode("ascii")
        result = audio_input(b64_str, format="mp3")
        assert result["type"] == "input_audio"
        assert result["input_audio"]["data"] == b64_str
        assert result["input_audio"]["format"] == "mp3"

    def test_audio_from_path(self, tmp_path: Path):
        mp3_file = tmp_path / "clip.mp3"
        mp3_file.write_bytes(b"\xff\xfb\x90\x00")  # MP3 sync word
        result = audio_from_path(mp3_file)
        assert result["type"] == "input_audio"
        assert result["input_audio"]["format"] == "mp3"
        # Verify data round-trips
        decoded = base64.b64decode(result["input_audio"]["data"])
        assert decoded == b"\xff\xfb\x90\x00"


# ---------------------------------------------------------------------------
# TestDocumentHelpers
# ---------------------------------------------------------------------------


class TestDocumentHelpers:
    def test_document_url(self):
        result = document_url("https://example.com/report.pdf")
        assert result == {
            "type": "document",
            "source": {
                "type": "url",
                "url": "https://example.com/report.pdf",
                "media_type": "application/pdf",
            },
        }

    def test_document_from_bytes(self):
        raw = b"%PDF-1.4 fake"
        result = document_from_bytes(raw)
        assert result["type"] == "document"
        source = result["source"]
        assert source["type"] == "base64"
        assert source["media_type"] == "application/pdf"
        assert base64.b64decode(source["data"]) == raw

    def test_document_from_path(self, tmp_path: Path):
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        result = document_from_path(pdf_file)
        assert result["type"] == "document"
        assert result["source"]["media_type"] == "application/pdf"
        assert base64.b64decode(result["source"]["data"]) == b"%PDF-1.4"


# ---------------------------------------------------------------------------
# TestContentPartMultimodal
# ---------------------------------------------------------------------------


class TestContentPartMultimodal:
    def test_image_part_with_binary(self):
        bd = BinaryData(data="aGVsbG8=", media_type="image/png")
        part = ContentPart(type="image", binary=bd)
        assert part.type == "image"
        assert part.binary is not None
        assert part.binary.is_image is True
        assert part.binary.data == "aGVsbG8="

    def test_audio_part_with_transcript(self):
        bd = BinaryData(data="AAAA", media_type="audio/wav")
        part = ContentPart(type="audio", binary=bd, transcript="hello")
        assert part.type == "audio"
        assert part.binary is not None
        assert part.binary.is_audio is True
        assert part.transcript == "hello"

    def test_document_part(self):
        bd = BinaryData(data="JVBER", media_type="application/pdf")
        part = ContentPart(type="document", binary=bd)
        assert part.type == "document"
        assert part.binary is not None
        assert part.binary.is_document is True


# ---------------------------------------------------------------------------
# TestProviderResponseMultimodal
# ---------------------------------------------------------------------------


class TestProviderResponseMultimodal:
    def _make_response(self, **overrides: Any) -> ProviderResponse:  # type: ignore[no-any-explicit]
        defaults: dict[str, Any] = {
            "provider": "test",
            "model": "test-model",
            "raw": {},
        }
        defaults.update(overrides)
        return ProviderResponse(**defaults)

    def test_images_property(self):
        img1 = BinaryData(data="aW1n", media_type="image/png")
        img2 = BinaryData(data="aW1nMg==", media_type="image/jpeg")
        resp = self._make_response(
            parts=[
                ContentPart(type="text", text="Here are two images"),
                ContentPart(type="image", binary=img1),
                ContentPart(type="image", binary=img2),
            ]
        )
        images = resp.images
        assert len(images) == 2
        assert images[0].media_type == "image/png"
        assert images[1].media_type == "image/jpeg"

    def test_audio_property(self):
        audio_bd = BinaryData(data="YXVkaW8=", media_type="audio/wav")
        resp = self._make_response(
            parts=[
                ContentPart(type="text", text="Here is audio"),
                ContentPart(type="audio", binary=audio_bd),
            ]
        )
        assert resp.audio is not None
        assert resp.audio.media_type == "audio/wav"
        assert resp.audio.data == "YXVkaW8="

    def test_audio_transcript_property(self):
        audio_bd = BinaryData(data="YXVkaW8=", media_type="audio/wav")
        resp = self._make_response(
            parts=[
                ContentPart(type="audio", binary=audio_bd, transcript="hello world"),
            ]
        )
        assert resp.audio_transcript == "hello world"

    def test_empty_multimodal_properties(self):
        resp = self._make_response(
            parts=[
                ContentPart(type="text", text="just text"),
            ]
        )
        assert resp.images == []
        assert resp.audio is None
        assert resp.audio_transcript is None
