from __future__ import annotations

import json
import math
import shutil
from copy import deepcopy
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from .config import Settings


DEFAULT_PAYLOAD = {"matches": {}, "videos": {}, "incidents": {}}


class JsonStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self.settings.storage_path.mkdir(parents=True, exist_ok=True)
        self.settings.media_path.mkdir(parents=True, exist_ok=True)
        if not self.settings.metadata_path.exists():
            self._write(self._default_payload())

    def _default_payload(self) -> dict[str, dict[str, Any]]:
        return deepcopy(DEFAULT_PAYLOAD)

    def _backup_path(self) -> Path:
        return self.settings.metadata_path.with_name(f"{self.settings.metadata_path.name}.bak")

    def _corrupt_snapshot_path(self) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return self.settings.storage_path / f"metadata.corrupt.{stamp}.json"

    def _read(self) -> dict[str, dict[str, Any]]:
        try:
            with self.settings.metadata_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            payload = self._default_payload()
            self._write(payload)
            return payload
        except JSONDecodeError:
            return self._recover_corrupted_metadata()

    def _write(self, payload: dict[str, dict[str, Any]]) -> None:
        normalized = self._normalize_payload(payload)
        serialized = json.dumps(normalized, indent=2, allow_nan=False)
        self._atomic_write(self.settings.metadata_path, serialized)
        self._atomic_write(self._backup_path(), serialized)

    def _atomic_write(self, path: Path, content: str) -> None:
        temp_path = path.with_name(f"{path.name}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
        temp_path.replace(path)

    def _normalize_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._normalize_payload(next_value) for key, next_value in value.items()}
        if isinstance(value, list):
            return [self._normalize_payload(item) for item in value]
        if isinstance(value, tuple):
            return [self._normalize_payload(item) for item in value]
        if isinstance(value, np.ndarray):
            return self._normalize_payload(value.tolist())
        if isinstance(value, np.generic):
            return self._normalize_payload(value.item())
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value

    def _recover_corrupted_metadata(self) -> dict[str, dict[str, Any]]:
        metadata_path = self.settings.metadata_path
        raw_text = metadata_path.read_text(encoding="utf-8", errors="ignore") if metadata_path.exists() else ""
        salvaged = self._salvage_payload(raw_text)
        backup_payload = self._read_backup()

        if self._score_payload(backup_payload) > self._score_payload(salvaged):
            recovered = backup_payload
        else:
            recovered = salvaged

        if metadata_path.exists():
            corrupt_snapshot = self._corrupt_snapshot_path()
            shutil.copyfile(metadata_path, corrupt_snapshot)

        self._write(recovered)
        return recovered

    def _read_backup(self) -> dict[str, dict[str, Any]]:
        backup_path = self._backup_path()
        if not backup_path.exists():
            return self._default_payload()
        try:
            with backup_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else self._default_payload()
        except (JSONDecodeError, OSError):
            return self._default_payload()

    def _score_payload(self, payload: dict[str, dict[str, Any]]) -> int:
        return sum(len(payload.get(bucket, {})) for bucket in DEFAULT_PAYLOAD)

    def _salvage_payload(self, raw_text: str) -> dict[str, dict[str, Any]]:
        payload = self._default_payload()
        for bucket in DEFAULT_PAYLOAD:
            recovered = self._extract_bucket(raw_text, bucket)
            if recovered is not None:
                payload[bucket] = recovered
        return payload

    def _extract_bucket(self, raw_text: str, bucket: str) -> dict[str, Any] | None:
        bucket_marker = f'"{bucket}"'
        marker_index = raw_text.find(bucket_marker)
        if marker_index == -1:
            return None

        start_index = raw_text.find("{", marker_index)
        if start_index == -1:
            return None

        bucket_text = self._extract_balanced_object(raw_text, start_index)
        if bucket_text is not None:
            try:
                parsed = json.loads(bucket_text)
                return parsed if isinstance(parsed, dict) else {}
            except JSONDecodeError:
                pass

        return self._extract_partial_records(raw_text, start_index)

    def _extract_balanced_object(self, raw_text: str, start_index: int) -> str | None:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start_index, len(raw_text)):
            char = raw_text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return raw_text[start_index : index + 1]
        return None

    def _extract_partial_records(self, raw_text: str, start_index: int) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        records: dict[str, Any] = {}
        cursor = start_index + 1
        while cursor < len(raw_text):
            while cursor < len(raw_text) and raw_text[cursor] in " \r\n\t,":
                cursor += 1
            if cursor >= len(raw_text) or raw_text[cursor] == "}":
                break
            if raw_text[cursor] != '"':
                break

            try:
                key, offset = decoder.raw_decode(raw_text[cursor:])
            except JSONDecodeError:
                break
            cursor += offset

            while cursor < len(raw_text) and raw_text[cursor] in " \r\n\t":
                cursor += 1
            if cursor >= len(raw_text) or raw_text[cursor] != ":":
                break
            cursor += 1

            while cursor < len(raw_text) and raw_text[cursor] in " \r\n\t":
                cursor += 1
            if cursor >= len(raw_text) or raw_text[cursor] != "{":
                break

            object_text = self._extract_balanced_object(raw_text, cursor)
            if object_text is None:
                break
            try:
                records[key] = json.loads(object_text)
            except JSONDecodeError:
                break
            cursor += len(object_text)

        return records

    def list_records(self, bucket: str) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read()
            return list(deepcopy(data[bucket]).values())

    def get_record(self, bucket: str, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._read()
            record = data[bucket].get(record_id)
            return deepcopy(record) if record else None

    def upsert_record(self, bucket: str, record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            data[bucket][record_id] = payload
            self._write(data)
            return deepcopy(payload)

    def delete_record(self, bucket: str, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._read()
            record = data[bucket].pop(record_id, None)
            if record is None:
                return None
            self._write(data)
            return deepcopy(record)

    def ensure_video(self, record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        existing = self.get_record("videos", record_id)
        return existing if existing else self.upsert_record("videos", record_id, payload)

    def media_dir(self, *parts: str) -> Path:
        path = self.settings.media_path.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def delete_media_dir(self, *parts: str) -> None:
        path = self.settings.media_path.joinpath(*parts)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
