from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np

from .config import get_settings
from .schemas import (
    AttackDirection,
    DemoLoginRequest,
    DemoLoginResponse,
    OffsideCorrectionRequest,
    GoalReviewRequest,
    IncidentNoteRequest,
    IncidentResponse,
    MatchCreateRequest,
    MatchResponse,
    OffsideFrameReviewRequest,
    ReviewClipRequest,
    ReviewClipResponse,
    VideoAssetResponse,
)
from .services.analysis import get_analyzer
from .services.media import (
    VIDEO_EXTENSIONS,
    create_poster,
    extract_clip,
    get_video_metadata,
    next_id,
    relative_media_url,
    save_upload_file,
)
from .storage import JsonStorage


settings = get_settings()
storage = JsonStorage(settings)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.media_path.mkdir(parents=True, exist_ok=True)
settings.sample_clips_path.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=settings.media_path), name="media")
app.mount("/samples", StaticFiles(directory=settings.sample_clips_path), name="samples")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def datetime_from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(next_value) for key, next_value in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def serialize_video(record: dict) -> VideoAssetResponse:
    return VideoAssetResponse(
        id=record["id"],
        name=record["name"],
        source_type=record["source_type"],
        duration=record["duration"],
        fps=record["fps"],
        width=record["width"],
        height=record["height"],
        url=record["url"],
        poster_url=record.get("poster_url"),
        created_at=datetime_from_iso(record["created_at"]),
    )


def serialize_match(record: dict) -> MatchResponse:
    return MatchResponse(
        id=record["id"],
        title=record["title"],
        home_team=record["home_team"],
        away_team=record["away_team"],
        kickoff=record["kickoff"],
        video_id=record["video_id"],
        status=record["status"],
        created_at=datetime_from_iso(record["created_at"]),
    )


def serialize_incident(record: dict) -> IncidentResponse:
    safe_record = json_safe(record)
    return IncidentResponse(
        id=safe_record["id"],
        match_id=safe_record.get("match_id"),
        video_id=safe_record["video_id"],
        review_type=safe_record["review_type"],
        review_timestamp=safe_record["review_timestamp"],
        clip_start=safe_record["clip_start"],
        clip_end=safe_record["clip_end"],
        clip_url=safe_record["clip_url"],
        source_video_url=safe_record.get("source_video_url"),
        status=safe_record["status"],
        verdict=safe_record.get("verdict"),
        confidence=safe_record.get("confidence"),
        frame_timestamp=safe_record.get("frame_timestamp"),
        frame_source_url=safe_record.get("frame_source_url"),
        snapshot_url=safe_record.get("snapshot_url"),
        diagram_url=safe_record.get("diagram_url"),
        note=safe_record.get("note"),
        rationale=safe_record.get("rationale"),
        diagnostics=safe_record.get("diagnostics", {}),
        suggestions=safe_record.get("suggestions", []),
        player_candidates=safe_record.get("player_candidates", []),
        selected_attacker_id=safe_record.get("selected_attacker_id"),
        selected_defender_id=safe_record.get("selected_defender_id"),
        attack_direction=safe_record.get("attack_direction"),
        created_at=datetime_from_iso(safe_record["created_at"]),
        updated_at=datetime_from_iso(safe_record["updated_at"]),
    )


def sync_sample_videos() -> list[dict]:
    records: list[dict] = []
    for file_path in sorted(settings.sample_clips_path.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        record_id = f"sample_{file_path.stem.lower().replace(' ', '_')}"
        meta = get_video_metadata(file_path)
        poster_path = settings.media_path / "samples" / record_id / "poster.jpg"
        poster_path.parent.mkdir(parents=True, exist_ok=True)
        if not poster_path.exists():
            create_poster(file_path, poster_path)
        record = {
            "id": record_id,
            "name": file_path.stem,
            "source_type": "sample",
            "duration": round(float(meta["duration"]), 2),
            "fps": round(float(meta["fps"]), 2),
            "width": int(meta["width"]),
            "height": int(meta["height"]),
            "path": str(file_path),
            "url": relative_media_url(file_path, settings),
            "poster_url": relative_media_url(poster_path, settings),
            "created_at": now_iso(),
        }
        records.append(storage.ensure_video(record_id, record))
    return records


@app.on_event("startup")
def on_startup() -> None:
    sync_sample_videos()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.post("/auth/demo-login", response_model=DemoLoginResponse)
def demo_login(payload: DemoLoginRequest) -> DemoLoginResponse:
    return DemoLoginResponse(
        session_id=next_id("session"),
        role=payload.role,
        display_name=payload.display_name,
        email=payload.email,
        created_at=datetime.now(UTC),
    )


@app.get("/sample-clips", response_model=list[VideoAssetResponse])
def list_sample_clips() -> list[VideoAssetResponse]:
    return [serialize_video(record) for record in sync_sample_videos()]


@app.post("/matches", response_model=MatchResponse)
def create_match(payload: MatchCreateRequest) -> MatchResponse:
    video = storage.get_record("videos", payload.video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video source not found.")
    record = {
        "id": next_id("match"),
        "title": payload.title,
        "home_team": payload.home_team,
        "away_team": payload.away_team,
        "kickoff": payload.kickoff,
        "video_id": payload.video_id,
        "status": "ready",
        "created_at": now_iso(),
    }
    storage.upsert_record("matches", record["id"], record)
    return serialize_match(record)


@app.post("/uploads/video", response_model=VideoAssetResponse)
async def upload_video(file: UploadFile = File(...)) -> VideoAssetResponse:
    suffix = Path(file.filename or "clip.mp4").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    video_id = next_id("video")
    destination_dir = storage.media_dir("videos", video_id)
    file_path = save_upload_file(file, destination_dir)
    metadata = get_video_metadata(file_path)
    poster_path = create_poster(file_path, destination_dir / "poster.jpg")
    record = {
        "id": video_id,
        "name": Path(file.filename or "uploaded clip").stem,
        "source_type": "upload",
        "duration": round(float(metadata["duration"]), 2),
        "fps": round(float(metadata["fps"]), 2),
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
        "path": str(file_path),
        "url": relative_media_url(file_path, settings),
        "poster_url": relative_media_url(poster_path, settings),
        "created_at": now_iso(),
    }
    storage.upsert_record("videos", video_id, record)
    return serialize_video(record)


@app.post("/reviews/clip", response_model=ReviewClipResponse)
def create_review_clip(payload: ReviewClipRequest) -> ReviewClipResponse:
    video = storage.get_record("videos", payload.video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video source not found.")
    duration = float(video["duration"])
    clip_length = 10.0 if payload.review_type == "offside" else 5.0
    clip_end = min(payload.review_timestamp, duration)
    clip_start = max(0.0, clip_end - clip_length)
    incident_id = next_id("incident")
    incident_dir = storage.media_dir("incidents", incident_id)
    clip_path = incident_dir / f"{payload.review_type}_clip.mp4"
    clip_meta = extract_clip(Path(video["path"]), clip_path, clip_start, clip_end)

    match_records = storage.list_records("matches")
    linked_match = next((record for record in reversed(match_records) if record["video_id"] == payload.video_id), None)
    record = {
        "id": incident_id,
        "match_id": linked_match["id"] if linked_match else None,
        "video_id": payload.video_id,
        "review_type": payload.review_type,
        "review_timestamp": payload.review_timestamp,
        "clip_start": clip_start,
        "clip_end": clip_end,
        "clip_url": relative_media_url(clip_path, settings),
        "clip_path": str(clip_path),
        "source_video_path": video["path"],
        "source_video_url": video["url"],
        "status": "pending_frame_lock" if payload.review_type == "offside" else "processing",
        "verdict": None,
        "confidence": None,
        "frame_timestamp": None,
        "frame_source_url": None,
        "snapshot_url": None,
        "diagram_url": None,
        "note": "",
        "rationale": None,
        "diagnostics": {},
        "suggestions": [],
        "player_candidates": [],
        "selected_attacker_id": None,
        "selected_defender_id": None,
        "attack_direction": None,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    storage.upsert_record("incidents", incident_id, record)
    return ReviewClipResponse(
        incident_id=incident_id,
        match_id=record["match_id"],
        review_type=payload.review_type,
        clip_start=clip_start,
        clip_end=clip_end,
        clip_duration=clip_meta["duration"],
        clip_url=record["clip_url"],
        source_video_url=record["source_video_url"],
        status=record["status"],
    )


@app.post("/reviews/offside/frame", response_model=IncidentResponse)
def review_offside_frame(payload: OffsideFrameReviewRequest) -> IncidentResponse:
    incident = storage.get_record("incidents", payload.incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found.")
    analyzer = get_analyzer(settings)
    source_path = Path(incident.get("source_video_path") or incident["clip_path"])
    result = analyzer.analyze_offside(source_path, storage.media_dir("incidents", incident["id"]), payload.frame_timestamp)
    incident.update(
        {
            "status": "reviewed",
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "frame_timestamp": result["frame_timestamp"],
            "frame_source_url": relative_media_url(result["frame_source_path"], settings) if result.get("frame_source_path") else None,
            "snapshot_url": relative_media_url(result["snapshot_path"], settings),
            "diagram_url": relative_media_url(result["diagram_path"], settings) if result["diagram_path"] else None,
            "rationale": result["rationale"],
            "diagnostics": result["diagnostics"],
            "suggestions": result["suggestions"],
            "player_candidates": result.get("player_candidates", []),
            "selected_attacker_id": result.get("selected_attacker_id"),
            "selected_defender_id": result.get("selected_defender_id"),
            "attack_direction": result.get("attack_direction"),
            "updated_at": now_iso(),
        }
    )
    incident = json_safe(incident)
    storage.upsert_record("incidents", incident["id"], incident)
    return serialize_incident(incident)


@app.post("/incidents/{incident_id}/offside-correction", response_model=IncidentResponse)
def apply_offside_correction(incident_id: str, payload: OffsideCorrectionRequest) -> IncidentResponse:
    incident = storage.get_record("incidents", incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found.")
    if incident.get("review_type") != "offside":
        raise HTTPException(status_code=400, detail="Manual correction is only available for offside reviews.")
    frame_timestamp = incident.get("frame_timestamp")
    if frame_timestamp is None:
        raise HTTPException(status_code=400, detail="Lock a frame before applying manual correction.")
    analyzer = get_analyzer(settings)
    source_path = Path(incident.get("source_video_path") or incident["clip_path"])
    result = analyzer.analyze_offside(
        source_path,
        storage.media_dir("incidents", incident["id"]),
        float(frame_timestamp),
        manual_selection={
            "attacker_id": payload.attacker_id,
            "defender_id": payload.defender_id,
            "attacker_point": payload.attacker_point,
            "defender_point": payload.defender_point,
            "attack_direction": payload.attack_direction,
        },
    )
    incident.update(
        {
            "status": "reviewed",
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "frame_timestamp": result["frame_timestamp"],
            "frame_source_url": relative_media_url(result["frame_source_path"], settings) if result.get("frame_source_path") else None,
            "snapshot_url": relative_media_url(result["snapshot_path"], settings),
            "diagram_url": relative_media_url(result["diagram_path"], settings) if result["diagram_path"] else None,
            "rationale": result["rationale"],
            "diagnostics": result["diagnostics"],
            "suggestions": result["suggestions"],
            "player_candidates": result.get("player_candidates", []),
            "selected_attacker_id": result.get("selected_attacker_id"),
            "selected_defender_id": result.get("selected_defender_id"),
            "attack_direction": result.get("attack_direction"),
            "updated_at": now_iso(),
        }
    )
    incident = json_safe(incident)
    storage.upsert_record("incidents", incident["id"], incident)
    return serialize_incident(incident)


@app.post("/reviews/goal/analyze", response_model=IncidentResponse)
def review_goal(payload: GoalReviewRequest) -> IncidentResponse:
    incident = storage.get_record("incidents", payload.incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found.")
    analyzer = get_analyzer(settings)
    result = analyzer.analyze_goal(Path(incident["clip_path"]), storage.media_dir("incidents", incident["id"]))
    incident.update(
        {
            "status": "reviewed",
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "frame_timestamp": result["frame_timestamp"],
            "frame_source_url": relative_media_url(result["frame_source_path"], settings) if result.get("frame_source_path") else None,
            "snapshot_url": relative_media_url(result["snapshot_path"], settings),
            "diagram_url": relative_media_url(result["diagram_path"], settings) if result.get("diagram_path") else None,
            "rationale": result["rationale"],
            "diagnostics": result["diagnostics"],
            "suggestions": result["suggestions"],
            "updated_at": now_iso(),
        }
    )
    incident = json_safe(incident)
    storage.upsert_record("incidents", incident["id"], incident)
    return serialize_incident(incident)


@app.post("/incidents/{incident_id}/note", response_model=IncidentResponse)
def save_incident_note(incident_id: str, payload: IncidentNoteRequest) -> IncidentResponse:
    incident = storage.get_record("incidents", incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found.")
    incident["note"] = payload.note
    incident["updated_at"] = now_iso()
    storage.upsert_record("incidents", incident_id, incident)
    return serialize_incident(incident)


@app.delete("/incidents/{incident_id}")
def delete_incident(incident_id: str) -> dict[str, str]:
    incident = storage.delete_record("incidents", incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found.")
    storage.delete_media_dir("incidents", incident_id)
    return {"id": incident_id, "status": "deleted"}


@app.get("/incidents", response_model=list[IncidentResponse])
def list_incidents() -> list[IncidentResponse]:
    incidents = sorted(storage.list_records("incidents"), key=lambda record: record["created_at"], reverse=True)
    return [serialize_incident(record) for record in incidents]


@app.get("/incidents/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: str) -> IncidentResponse:
    incident = storage.get_record("incidents", incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found.")
    return serialize_incident(incident)
