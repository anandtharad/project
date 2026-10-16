"""Raw PCM from streamed audio that may carry WAV headers.

Providers that stream WAV may send one header at the start (possibly split
across chunks, possibly in a chunk of its own) or a whole WAV file per chunk.
`WavStripper` handles all three: a chunk that starts with "RIFF" is parsed as
a header, an incomplete header is held until the rest arrives, and the
header's format is checked against the format announced to the client, so a
mismatch fails clearly instead of playing at the wrong speed.
"""

from __future__ import annotations

import struct

from app.voice.base import AudioFormat, SpeechError

# A header longer than this is not a header we can use.
MAX_HEADER_BYTES = 4096
_PCM = 1
_EXTENSIBLE = 0xFFFE


class WavStripper:
    def __init__(self, provider: str, expected: AudioFormat) -> None:
        self._provider = provider
        self._expected = expected
        self._pending = b""

    def feed(self, data: bytes) -> bytes:
        """The PCM in this chunk (possibly empty while a header is incomplete)."""
        if self._pending:
            data, self._pending = self._pending + data, b""
        elif not data.startswith(b"RIFF"):
            return data
        start = self._data_offset(data)
        if start is None:
            if len(data) > MAX_HEADER_BYTES:
                raise SpeechError(self._provider, "bad_wav_header")
            self._pending = data
            return b""
        return data[start:]

    def _data_offset(self, data: bytes) -> int | None:
        """Where the samples start, or None if the header is not complete yet."""
        if len(data) < 12:
            return None
        if data[8:12] != b"WAVE":
            raise SpeechError(self._provider, "bad_wav_header")
        offset = 12
        while len(data) >= offset + 8:
            chunk_id = data[offset : offset + 4]
            size = struct.unpack_from("<I", data, offset + 4)[0]
            if chunk_id == b"data":
                return offset + 8
            if chunk_id == b"fmt ":
                if len(data) < offset + 8 + 16:
                    return None
                self._check_format(data, offset + 8)
            offset += 8 + size + (size & 1)
        return None

    def _check_format(self, data: bytes, at: int) -> None:
        tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", data, at)
        expected = self._expected
        if (
            tag not in (_PCM, _EXTENSIBLE)
            or bits != 16
            or channels != expected.channels
            or rate != expected.sample_rate
        ):
            raise SpeechError(
                self._provider, f"unexpected_audio_format:{tag}/{bits}bit/{channels}ch/{rate}Hz"
            )
