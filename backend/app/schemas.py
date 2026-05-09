from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# These literals keep the API contract narrow and predictable for the frontend.
Role = Literal["Match Official", "Team Viewer"]
ReviewType = Literal["offside", "goal"]
ReviewStatus = Literal["pending_frame_lock", "processing", "reviewed"]
Verdict = Literal["Offside", "Onside", "Goal", "No Goal", "Human Review"]
AttackDirection = Literal["left", "right"]


# Demo auth is intentionally lightweight; this is just enough to drive role-based UI states.
class DemoLoginRequest(BaseModel):
    role: Role
    display_name: str = Field(min_length=2, max_length=60)
    email: str | None = None


class DemoLoginResponse(BaseModel):
    session_id: str
    role: Role
    display_name: str
    email: str | None = None
    created_at: datetime


# Match metadata is stored separately from video assets so one source clip can back a fixture.
class MatchCreateRequest(BaseModel):
    title: str = Field(min_length=2, max_length=120)
    home_team: str = Field(min_length=2, max_length=60)
    away_team: str = Field(min_length=2, max_length=60)
    kickoff: str
    video_id: str


class MatchResponse(BaseModel):
    id: str
    title: str
    home_team: str
    away_team: str
    kickoff: str
    video_id: str
    status: str
    created_at: datetime


class VideoAssetResponse(BaseModel):
    id: str
    name: str
    source_type: Literal["upload", "sample"]
    duration: float
    fps: float
    width: int
    height: int
    url: str
    poster_url: str | None = None
    created_at: datetime


# A review clip is the short working window generated from the longer source feed.
class ReviewClipRequest(BaseModel):
    video_id: str
    review_type: ReviewType
    review_timestamp: float = Field(ge=0)


class ReviewClipResponse(BaseModel):
    incident_id: str
    match_id: str | None = None
    review_type: ReviewType
    clip_start: float
    clip_end: float
    clip_duration: float
    clip_url: str
    source_video_url: str | None = None
    status: ReviewStatus


class OffsideFrameReviewRequest(BaseModel):
    incident_id: str
    frame_timestamp: float = Field(ge=0)


class GoalReviewRequest(BaseModel):
    incident_id: str


# Suggestions are simple labels used by the UI when a review result needs operator context.
class DetectionSuggestion(BaseModel):
    id: str
    label: str
    team: str | None = None
    confidence: float
    bbox: list[float]


# Player candidates are richer because manual offside correction needs feet and pitch context.
class PlayerCandidate(BaseModel):
    id: str
    team: str | None = None
    confidence: float
    bbox: list[float]
    feet_point: list[float]
    pitch_score: float | None = None
    on_pitch: bool | None = None


# This is the main record the frontend works with after a review starts.
class IncidentResponse(BaseModel):
    id: str
    match_id: str | None = None
    video_id: str
    review_type: ReviewType
    review_timestamp: float
    clip_start: float
    clip_end: float
    clip_url: str
    source_video_url: str | None = None
    status: ReviewStatus
    verdict: Verdict | None = None
    confidence: float | None = None
    frame_timestamp: float | None = None
    frame_source_url: str | None = None
    snapshot_url: str | None = None
    diagram_url: str | None = None
    note: str | None = None
    rationale: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    suggestions: list[DetectionSuggestion] = Field(default_factory=list)
    player_candidates: list[PlayerCandidate] = Field(default_factory=list)
    selected_attacker_id: str | None = None
    selected_defender_id: str | None = None
    attack_direction: AttackDirection | None = None
    created_at: datetime
    updated_at: datetime


# Notes are optional, but bounded so they stay short and readable in the incident archive.
class IncidentNoteRequest(BaseModel):
    note: str = Field(min_length=0, max_length=300)


# Manual offside correction can use detected player ids, manually planted points, or both.
class OffsideCorrectionRequest(BaseModel):
    attacker_id: str | None = None
    defender_id: str | None = None
    attacker_point: list[float] | None = None
    defender_point: list[float] | None = None
    attack_direction: AttackDirection
