from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from ..config import Settings
from .media import extract_frame, sample_video_timestamps


SPORTS_BALL_CLASS = 32
PERSON_CLASS = 0


@dataclass
class Player:
    id: str
    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: np.ndarray | None
    team: str | None = None
    jersey_color: tuple[int, int, int] | None = None
    pitch_score: float = 0.0
    on_pitch: bool = True

    @property
    def feet_point(self) -> tuple[float, float]:
        if self.keypoints is not None and self.keypoints.shape[0] >= 17:
            ankles = self.keypoints[[15, 16]]
            valid = ankles[np.all(ankles > 0, axis=1)]
            if len(valid):
                return float(valid[:, 0].mean()), float(valid[:, 1].mean())
        x1, _, x2, y2 = self.bbox
        return (x1 + x2) / 2, y2

    @property
    def torso_crop(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox
        height = y2 - y1
        return int(x1), int(y1 + height * 0.15), int(x2), int(y1 + height * 0.55)


class VisionAnalyzer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._detect_model: YOLO | None = None
        self._pose_model: YOLO | None = None
        self.device = self._resolve_device()

    def _resolve_device(self) -> str:
        if self.settings.model_device != "auto":
            return self.settings.model_device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _detect(self) -> YOLO:
        if self._detect_model is None:
            self._detect_model = YOLO("yolo11n.pt")
        return self._detect_model

    def _pose(self) -> YOLO:
        if self._pose_model is None:
            self._pose_model = YOLO("yolo11n-pose.pt")
        return self._pose_model

    def analyze_offside(
        self,
        video_path: Path,
        output_dir: Path,
        frame_timestamp: float,
        manual_selection: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        frame, actual_time = extract_frame(video_path, frame_timestamp)
        frame_source_path = output_dir / "offside_frame.jpg"
        cv2.imwrite(str(frame_source_path), frame)
        try:
            detect_result = self._detect()(frame, device=self.device, conf=0.12, imgsz=960, verbose=False)[0]
            pose_result = self._pose()(frame, device=self.device, conf=0.18, imgsz=960, verbose=False)[0]
        except Exception as exc:
            snapshot_path = output_dir / "offside_snapshot.jpg"
            cv2.imwrite(str(snapshot_path), frame)
            return {
                "verdict": "Human Review",
                "confidence": 0.12,
                "frame_timestamp": actual_time,
                "frame_source_path": frame_source_path,
                "snapshot_path": snapshot_path,
                "diagram_path": None,
                "rationale": "The review engine could not complete model inference for the locked frame.",
                "diagnostics": {"device": self.device, "model_error": str(exc)},
                "suggestions": [],
                "player_candidates": [],
                "selected_attacker_id": None,
                "selected_defender_id": None,
                "attack_direction": None,
            }

        geometry = self._estimate_pitch_geometry(frame)
        players = self._filter_pitch_players(self._extract_players(detect_result, pose_result), geometry, frame.shape)
        ball = self._extract_ball(detect_result)
        if ball is not None and not self._point_is_on_pitch(ball["center"], geometry, frame.shape, radius=28):
            ball = None
        future_balls = self._track_future_ball_positions(video_path, actual_time)
        self._cluster_teams(frame, players)
        player_candidates = [self._player_candidate(player) for player in players]

        diagnostics = {
            "line_strength": round(geometry["line_strength"], 3),
            "ball_detected": bool(ball is not None),
            "future_ball_samples": len(future_balls),
            "player_count": len(players),
            "device": self.device,
            "pitch_player_count": len(players),
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
        }

        verdict = "Human Review"
        confidence = 0.2
        rationale = "The frame does not expose enough clean positional data to force a call."
        suggestions: list[dict[str, Any]] = []
        diagram_path: Path | None = None
        selected_attacker_id: str | None = None
        selected_defender_id: str | None = None
        selected_attack_direction: str | None = None

        if manual_selection and len(players) >= 2:
            manual_context = self._resolve_manual_offside_context(players, manual_selection)
            if manual_context is not None:
                attacker, defender, attack_direction = manual_context
                return self._finalize_offside_result(
                    frame=frame,
                    output_dir=output_dir,
                    actual_time=actual_time,
                    geometry=geometry,
                    ball=ball,
                    future_balls=future_balls,
                    players=players,
                    attacker=attacker,
                    defender=defender,
                    attack_direction=attack_direction,
                    diagnostics=diagnostics,
                    player_candidates=player_candidates,
                    locked_passer=None,
                    context_mode="manual",
                    selection_mode="manual",
                )

        if len(players) >= 4:
            context = self._resolve_offside_context(players, ball, future_balls, geometry, frame.shape)
            if context is not None:
                passer, receiver, second_last_defender, attack_direction, context_mode = context
                locked_passer = passer if context_mode == "ball" else None
                return self._finalize_offside_result(
                    frame=frame,
                    output_dir=output_dir,
                    actual_time=actual_time,
                    geometry=geometry,
                    ball=ball,
                    future_balls=future_balls,
                    players=players,
                    attacker=receiver,
                    defender=second_last_defender,
                    attack_direction=attack_direction,
                    diagnostics=diagnostics,
                    player_candidates=player_candidates,
                    locked_passer=locked_passer,
                    context_mode=context_mode,
                    selection_mode="auto",
                )
            diagnostics.update({"team_split": "insufficient"})

        snapshot_path = output_dir / "offside_snapshot.jpg"
        cv2.imwrite(str(snapshot_path), frame)
        return {
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "frame_timestamp": actual_time,
            "frame_source_path": frame_source_path,
            "snapshot_path": snapshot_path,
            "diagram_path": diagram_path,
            "rationale": rationale,
            "diagnostics": diagnostics,
            "suggestions": suggestions,
            "player_candidates": player_candidates,
            "selected_attacker_id": selected_attacker_id,
            "selected_defender_id": selected_defender_id,
            "attack_direction": selected_attack_direction,
        }

    def analyze_goal(self, video_path: Path, output_dir: Path) -> dict[str, Any]:
        metadata_frame, _ = extract_frame(video_path, 0)
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        capture.release()
        clip_duration = float(frame_count / fps) if fps else 5.0
        timestamps = sample_video_timestamps(clip_duration or 5.0, 10, focus_end=True)

        samples: list[dict[str, Any]] = []
        for timestamp in timestamps:
            frame, actual = extract_frame(video_path, timestamp)
            try:
                detect_result = self._detect()(frame, device=self.device, conf=0.12, imgsz=960, verbose=False)[0]
            except Exception as exc:
                snapshot_path = output_dir / "goal_snapshot.jpg"
                cv2.imwrite(str(snapshot_path), metadata_frame)
                return {
                    "verdict": "Human Review",
                    "confidence": 0.12,
                    "frame_timestamp": None,
                    "frame_source_path": None,
                    "snapshot_path": snapshot_path,
                    "diagram_path": None,
                    "rationale": "The review engine could not complete model inference for the goal sequence.",
                    "diagnostics": {"device": self.device, "model_error": str(exc)},
                    "suggestions": [],
                }
            ball = self._extract_ball(detect_result)
            if ball:
                geometry = self._estimate_pitch_geometry(frame)
                samples.append({"timestamp": actual, "frame": frame, "ball": ball, "geometry": geometry})

        if not samples:
            snapshot_path = output_dir / "goal_snapshot.jpg"
            cv2.imwrite(str(snapshot_path), metadata_frame)
            return {
                "verdict": "Human Review",
                "confidence": 0.19,
                "frame_timestamp": None,
                "frame_source_path": None,
                "snapshot_path": snapshot_path,
                "diagram_path": None,
                "rationale": "The scan could not keep the ball in view across the goal sequence.",
                "diagnostics": {"device": self.device, "sample_count": 0},
                "suggestions": [],
            }

        best = max(samples, key=lambda sample: sample["ball"]["confidence"])
        goal_side = "right" if mean(sample["ball"]["center"][0] for sample in samples) >= best["frame"].shape[1] / 2 else "left"
        evaluated_samples: list[dict[str, Any]] = []
        for sample in samples:
            sample_line_x = sample["geometry"]["goal_line_right_x"] if goal_side == "right" else sample["geometry"]["goal_line_left_x"]
            if sample_line_x is None:
                continue
            clearance_px = self._goal_line_clearance(sample["ball"], sample_line_x, goal_side)
            clear_margin_px = self._goal_clearance_margin(sample["ball"], sample["frame"].shape)
            boundary_state = self._goal_boundary_state(clearance_px, clear_margin_px)
            evaluated_samples.append(
                {
                    **sample,
                    "line_x": sample_line_x,
                    "clearance_px": clearance_px,
                    "clear_margin_px": clear_margin_px,
                    "boundary_state": boundary_state,
                }
            )

        line_found = len(evaluated_samples) > 0
        clear_frames = sum(1 for sample in evaluated_samples if sample["boundary_state"] == "clear")
        touching_frames = sum(1 for sample in evaluated_samples if sample["boundary_state"] == "touching")
        overlap_frames = sum(1 for sample in evaluated_samples if sample["boundary_state"] == "overlap")

        render_sample = (
            max(
                evaluated_samples,
                key=lambda sample: (
                    1 if sample["boundary_state"] == "clear" else 0,
                    sample["clearance_px"],
                    sample["geometry"]["line_strength"],
                    sample["ball"]["confidence"],
                ),
            )
            if clear_frames
            else min(
                evaluated_samples,
                key=lambda sample: (
                    abs(sample["clearance_px"]),
                    -sample["geometry"]["line_strength"],
                    -sample["ball"]["confidence"],
                ),
            )
            if evaluated_samples
            else best
        )
        line_x = render_sample.get("line_x")
        boundary_state = render_sample.get("boundary_state", "unknown")
        clearance_px = float(render_sample.get("clearance_px", 0.0))
        clear_margin_px = float(render_sample.get("clear_margin_px", 0.0))

        confidence = min(
            0.94,
            0.25
            + render_sample["geometry"]["line_strength"] * 0.28
            + min(0.25, len(samples) * 0.03)
            + min(0.18, render_sample["ball"]["confidence"] * 0.18),
        )
        if line_found and confidence >= 0.56:
            if boundary_state == "clear":
                verdict = "Goal"
                rationale = (
                    "The whole ball clears the projected goal line with visible separation in multiple sampled frames."
                    if clear_frames >= 2
                    else "The strongest sampled frame shows the whole ball clearly past the projected goal line."
                )
            else:
                verdict = "No Goal"
                if boundary_state == "touching":
                    rationale = "The ball still touches the projected goal line in the closest sampled frame, so this remains no goal."
                else:
                    rationale = "The ball still overlaps the projected goal line in the sampled frames and never fully clears it."
        else:
            verdict = "Human Review"
            rationale = "The sampled frames show a goalmouth event, but the goal line cannot be projected with enough confidence."

        frame_source_path = output_dir / "goal_frame.jpg"
        cv2.imwrite(str(frame_source_path), render_sample["frame"])
        snapshot_path = output_dir / "goal_snapshot.jpg"
        diagram_path = output_dir / "goal_diagram.png"
        annotated = self._draw_goal_overlay(
            render_sample["frame"],
            render_sample["ball"],
            line_x,
            goal_side,
            verdict,
            boundary_state,
            clearance_px,
            clear_margin_px,
        )
        cv2.imwrite(str(snapshot_path), annotated)
        self._draw_goal_diagram(diagram_path, goal_side, verdict, boundary_state, clearance_px, clear_margin_px)
        return {
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "frame_timestamp": render_sample["timestamp"],
            "frame_source_path": frame_source_path,
            "snapshot_path": snapshot_path,
            "diagram_path": diagram_path,
            "rationale": rationale,
            "diagnostics": {
                "device": self.device,
                "goal_side": goal_side,
                "sample_count": len(samples),
                "crossings": clear_frames,
                "touching_frames": touching_frames,
                "overlap_frames": overlap_frames,
                "boundary_state": boundary_state,
                "clearance_px": round(clearance_px, 2),
                "clear_margin_px": round(clear_margin_px, 2),
                "line_strength": round(render_sample["geometry"]["line_strength"], 3),
                "line_found": line_found,
                "ball_detected": True,
                "decision_mode": "strongest_clear_frame" if boundary_state == "clear" and clear_frames == 1 else "multi_sample",
            },
            "suggestions": [],
        }

    def _extract_players(self, detect_result: Any, pose_result: Any) -> list[Player]:
        players: list[Player] = []
        if pose_result.boxes is not None:
            boxes = pose_result.boxes.xyxy.cpu().numpy()
            scores = pose_result.boxes.conf.cpu().numpy()
            keypoints = pose_result.keypoints.xy.cpu().numpy() if pose_result.keypoints is not None else None
            for index, (bbox, score) in enumerate(zip(boxes, scores, strict=False)):
                if float(score) < 0.18:
                    continue
                player_keypoints = keypoints[index] if keypoints is not None and index < len(keypoints) else None
                players.append(Player(id=f"pose_player_{index}", bbox=tuple(map(float, bbox)), confidence=float(score), keypoints=player_keypoints))

        if detect_result.boxes is None:
            return players

        boxes = detect_result.boxes.xyxy.cpu().numpy()
        classes = detect_result.boxes.cls.cpu().numpy()
        scores = detect_result.boxes.conf.cpu().numpy()
        detect_index = 0
        for bbox, cls, score in zip(boxes, classes, scores, strict=False):
            if int(cls) != PERSON_CLASS or float(score) < 0.16:
                continue
            bbox_tuple = tuple(map(float, bbox))
            if any(self._bbox_iou(existing.bbox, bbox_tuple) >= 0.4 for existing in players):
                continue
            players.append(Player(id=f"detect_player_{detect_index}", bbox=bbox_tuple, confidence=float(score) * 0.92, keypoints=None))
            detect_index += 1
        return players

    def _extract_ball(self, detect_result: Any) -> dict[str, Any] | None:
        if detect_result.boxes is None:
            return None
        best: dict[str, Any] | None = None
        boxes = detect_result.boxes.xyxy.cpu().numpy()
        classes = detect_result.boxes.cls.cpu().numpy()
        scores = detect_result.boxes.conf.cpu().numpy()
        for bbox, cls, score in zip(boxes, classes, scores, strict=False):
            if int(cls) != SPORTS_BALL_CLASS:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            candidate = {
                "bbox": [x1, y1, x2, y2],
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                "radius": max((x2 - x1) / 2, (y2 - y1) / 2),
                "confidence": float(score),
            }
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate
        return best

    def _estimate_pitch_geometry(self, frame: np.ndarray) -> dict[str, Any]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([28, 35, 35]), np.array([96, 255, 255]))
        pitch_region = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        pitch_region = cv2.morphologyEx(pitch_region, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        pitch_region = cv2.dilate(pitch_region, np.ones((9, 9), np.uint8), iterations=1)
        pitch_hull = self._build_pitch_hull_mask(pitch_region)
        pitch_core = cv2.erode(pitch_hull, np.ones((17, 17), np.uint8), iterations=2)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([180, 70, 255]))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        white_mask = cv2.bitwise_and(white_mask, pitch_hull)
        edges = cv2.Canny(white_mask, 50, 140)
        white_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=55, minLineLength=max(60, frame.shape[1] // 10), maxLineGap=28)

        value_channel = cv2.GaussianBlur(hsv[..., 2], (7, 7), 0)
        stripe_edges = cv2.Canny(value_channel, 24, 72)
        stripe_edges = cv2.bitwise_and(stripe_edges, pitch_core)
        stripe_lines = cv2.HoughLinesP(
            stripe_edges,
            1,
            np.pi / 180,
            threshold=42,
            minLineLength=max(80, frame.shape[1] // 8),
            maxLineGap=34,
        )

        vertical_candidates: list[tuple[float, float, float, tuple[int, int, int, int], str]] = []
        vertical_candidates.extend(self._collect_parallel_line_candidates(white_lines, pitch_core, frame.shape, "white"))
        vertical_candidates.extend(self._collect_parallel_line_candidates(stripe_lines, pitch_core, frame.shape, "stripe"))
        vanishing_point = self._estimate_vanishing_point(vertical_candidates, frame.shape)
        left_goal_x = self._select_goalpost_anchor(vertical_candidates, frame.shape, "left")
        right_goal_x = self._select_goalpost_anchor(vertical_candidates, frame.shape, "right")

        if vertical_candidates:
            _, best_angle, best_length, best_line, _ = max(vertical_candidates, key=lambda item: item[0])
            x1, y1, x2, y2 = best_line
            if y2 < y1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            best_direction = self._normalize_direction(float(x2 - x1), float(y2 - y1))
            line_direction = self._weighted_parallel_direction(vertical_candidates)
            if line_direction is not None and best_direction is not None:
                blended = self._normalize_direction(
                    line_direction[0] * 0.62 + best_direction[0] * 0.38,
                    line_direction[1] * 0.62 + best_direction[1] * 0.38,
                )
                if blended is not None and abs(blended[0]) >= abs(best_direction[0]) * 0.78:
                    line_direction = blended
                else:
                    line_direction = best_direction
            elif line_direction is None:
                line_direction = best_direction
            projection_direction = best_direction if best_direction is not None else line_direction
            slope = 0.0 if projection_direction is None else projection_direction[0] / max(1e-6, projection_direction[1])
        else:
            best_angle = 90.0
            best_length = 0.0
            slope = 0.0
            line_direction = None

        line_strength = min(1.0, (best_length / max(1.0, frame.shape[0] * 0.55)) * 0.7 + len(vertical_candidates) / 10.0)
        return {
            "dominant_angle": best_angle,
            "slope": slope,
            "line_strength": line_strength,
            "goal_line_left_x": left_goal_x,
            "goal_line_right_x": right_goal_x,
            "reference_line": best_line if vertical_candidates else None,
            "line_direction": line_direction,
            "candidate_lines": vertical_candidates,
            "vanishing_point": vanishing_point,
            "pitch_region": pitch_hull,
            "pitch_core": pitch_core,
        }

    def _select_goalpost_anchor(
        self,
        candidates: list[tuple[float, float, float, tuple[int, int, int, int], str]],
        frame_shape: tuple[int, int, int],
        side: str,
    ) -> float | None:
        width = frame_shape[1]
        height = frame_shape[0]
        scored: list[tuple[float, float]] = []
        for score, _, length, line, source in candidates:
            if source != "white":
                continue
            x1, y1, x2, y2 = line
            x_mid = (x1 + x2) / 2
            y_top = min(y1, y2)
            y_bottom = max(y1, y2)
            vertical_span = (y_bottom - y_top) / max(1.0, height)
            if vertical_span < 0.18:
                continue
            if side == "left":
                if x_mid > width * 0.42:
                    continue
                front_bias = x_mid / max(1.0, width)
            else:
                if x_mid < width * 0.58:
                    continue
                front_bias = (width - x_mid) / max(1.0, width)
            top_bias = 1.0 - min(0.65, y_top / max(1.0, height))
            candidate_score = score * 0.55 + length * 0.25 + vertical_span * 180.0 + front_bias * 160.0 + top_bias * 36.0
            scored.append((candidate_score, x_mid))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return float(scored[0][1])

    def _collect_parallel_line_candidates(
        self,
        lines: np.ndarray | None,
        pitch_mask: np.ndarray,
        frame_shape: tuple[int, int, int],
        source: str,
    ) -> list[tuple[float, float, float, tuple[int, int, int, int], str]]:
        if lines is None:
            return []
        candidates: list[tuple[float, float, float, tuple[int, int, int, int], str]] = []
        width = frame_shape[1]
        for raw_line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, raw_line)
            length = math.hypot(x2 - x1, y2 - y1)
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if not (55 <= angle <= 125):
                continue
            pitch_ratio = self._line_pitch_ratio(pitch_mask, x1, y1, x2, y2)
            if pitch_ratio < (0.68 if source == "white" else 0.82):
                continue
            x_mid = (x1 + x2) / 2
            central_bias = 1.0 - min(0.42, abs(x_mid - width / 2) / width)
            source_weight = 1.0 if source == "white" else 0.62
            score = length * source_weight * (0.75 + pitch_ratio * 0.55 + central_bias * 0.18)
            candidates.append((score, angle, length, (x1, y1, x2, y2), source))
        return candidates

    def _weighted_parallel_direction(
        self,
        candidates: list[tuple[float, float, float, tuple[int, int, int, int], str]],
    ) -> tuple[float, float] | None:
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        weighted_dx = 0.0
        weighted_dy = 0.0
        total_weight = 0.0
        for score, _, _, line, _ in ranked[: min(8, len(ranked))]:
            x1, y1, x2, y2 = line
            direction = self._normalize_direction(float(x2 - x1), float(y2 - y1))
            if direction is None:
                continue
            weighted_dx += direction[0] * score
            weighted_dy += direction[1] * score
            total_weight += score
        if total_weight <= 1e-6:
            return None
        return self._normalize_direction(weighted_dx / total_weight, weighted_dy / total_weight)

    def _estimate_vanishing_point(
        self,
        candidates: list[tuple[float, float, float, tuple[int, int, int, int], str]],
        frame_shape: tuple[int, int, int],
    ) -> tuple[float, float] | None:
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        if len(ranked) < 2:
            return None

        rows: list[list[float]] = []
        values: list[float] = []
        weights: list[float] = []
        for score, _, _, line, _ in ranked[: min(10, len(ranked))]:
            x1, y1, x2, y2 = line
            a = float(y1 - y2)
            b = float(x2 - x1)
            magnitude = math.hypot(a, b)
            if magnitude <= 1e-6:
                continue
            a /= magnitude
            b /= magnitude
            c = -a * float(x1) - b * float(y1)
            rows.append([a, b])
            values.append(-c)
            weights.append(max(1.0, score))

        if len(rows) < 2:
            return None

        A = np.array(rows, dtype=np.float32)
        b = np.array(values, dtype=np.float32)
        W = np.sqrt(np.array(weights, dtype=np.float32))[:, None]
        try:
            solution, *_ = np.linalg.lstsq(A * W, b * W[:, 0], rcond=None)
        except np.linalg.LinAlgError:
            return None

        x, y = float(solution[0]), float(solution[1])
        width = frame_shape[1]
        height = frame_shape[0]
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        if x < -width * 3 or x > width * 4 or y < -height * 4 or y > height * 4:
            return None
        return x, y

    def _build_pitch_hull_mask(self, pitch_region: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(pitch_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return pitch_region
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(pitch_region)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        return mask

    def _filter_pitch_players(
        self,
        players: list[Player],
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> list[Player]:
        if not players:
            return []
        scored: list[tuple[float, Player]] = []
        for player in players:
            score = self._player_pitch_score(player, geometry, frame_shape)
            player.pitch_score = score
            player.on_pitch = score >= 0.46
            scored.append((score, player))

        filtered = [player for score, player in scored if score >= 0.46]
        if len(filtered) >= 4:
            return filtered

        relaxed_filtered = [player for score, player in scored if score >= 0.3]
        if len(relaxed_filtered) >= 4:
            return relaxed_filtered

        ranked = [player for _, player in sorted(scored, key=lambda item: item[0], reverse=True)]
        return ranked[: min(len(ranked), max(4, len(filtered)))]

    def _player_pitch_score(
        self,
        player: Player,
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> float:
        x1, y1, x2, y2 = player.bbox
        width = frame_shape[1]
        height = frame_shape[0]
        feet_x, feet_y = player.feet_point
        if x2 <= 4 or x1 >= width - 4 or y2 <= 6 or y1 >= height - 4:
            return 0.0

        score = 0.08
        row_bounds = self._pitch_row_bounds(geometry, int(round(feet_y)), width)
        if row_bounds is not None:
            left_bound, right_bound = row_bounds
            margin = 10.0
            if feet_x < left_bound - margin or feet_x > right_bound + margin:
                return 0.03
            score += 0.12

        feet_on_surface = self._point_is_on_pitch((feet_x, feet_y), geometry, frame_shape, radius=14, mask_key="pitch_region", threshold=72.0)
        lower_anchor = ((x1 + x2) / 2, y2 - max(10.0, (y2 - y1) * 0.14))
        lower_on_core = self._point_is_on_pitch(lower_anchor, geometry, frame_shape, radius=16, mask_key="pitch_core", threshold=82.0)
        lower_overlap_core = self._bbox_pitch_overlap(player, geometry, frame_shape, mask_key="pitch_core", lower_portion=0.48)
        lower_overlap_surface = self._bbox_pitch_overlap(player, geometry, frame_shape, mask_key="pitch_region", lower_portion=0.48)

        score += 0.34 if feet_on_surface else 0.0
        score += 0.18 if lower_on_core else 0.0
        score += min(0.16, lower_overlap_surface * 0.22)
        score += min(0.14, lower_overlap_core * 0.28)

        if not lower_on_core and lower_overlap_surface < 0.26:
            score *= 0.35
        if lower_overlap_core < 0.08 and lower_overlap_surface < 0.28:
            score *= 0.28

        return round(max(0.0, min(1.0, score)), 3)

    def _player_is_on_pitch(
        self,
        player: Player,
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> bool:
        return self._player_pitch_score(player, geometry, frame_shape) >= 0.46

    def _point_is_on_pitch(
        self,
        point: tuple[float, float],
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
        radius: int = 18,
        mask_key: str = "pitch_region",
        threshold: float = 70.0,
    ) -> bool:
        pitch_region = geometry.get(mask_key)
        if pitch_region is None:
            pitch_region = geometry.get("pitch_region")
        if pitch_region is None:
            return True
        x, y = point
        width = frame_shape[1]
        height = frame_shape[0]
        if x < -2 or x > width + 2 or y < -2 or y > height + 2:
            return False
        x_center = int(round(min(width - 1, max(0.0, x))))
        y_center = int(round(min(height - 1, max(0.0, y))))
        x_start = max(0, x_center - radius)
        x_end = min(width, x_center + radius + 1)
        y_start = max(0, y_center - radius)
        y_end = min(height, y_center + radius + 1)
        patch = pitch_region[y_start:y_end, x_start:x_end]
        if patch.size == 0:
            return False
        return float(patch.mean()) >= threshold

    def _bbox_pitch_overlap(
        self,
        player: Player,
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
        mask_key: str = "pitch_region",
        lower_portion: float = 0.5,
    ) -> float:
        pitch_region = geometry.get(mask_key)
        if pitch_region is None:
            pitch_region = geometry.get("pitch_region")
        if pitch_region is None:
            return 1.0
        x1, y1, x2, y2 = player.bbox
        width = frame_shape[1]
        height = frame_shape[0]
        x_start = max(0, int(round(x1)))
        x_end = min(width, int(round(x2)))
        y_start = max(0, int(round(y1 + (y2 - y1) * (1.0 - lower_portion))))
        y_end = min(height, int(round(y2)))
        patch = pitch_region[y_start:y_end, x_start:x_end]
        if patch.size == 0:
            return 0.0
        return float(patch.mean()) / 255.0

    def _pitch_row_bounds(
        self,
        geometry: dict[str, Any],
        row_y: int,
        width: int,
    ) -> tuple[int, int] | None:
        pitch_region = geometry.get("pitch_region")
        if pitch_region is None:
            return None
        row = int(round(min(pitch_region.shape[0] - 1, max(0, row_y))))
        active = np.where(pitch_region[row] > 0)[0]
        if len(active) == 0:
            return None
        return int(active[0]), int(active[-1])

    def _cluster_teams(self, frame: np.ndarray, players: list[Player]) -> None:
        features: list[np.ndarray] = []
        feature_players: list[Player] = []
        feature_map: dict[str, np.ndarray] = {}

        for player in players:
            player.team = None
            feature, color = self._extract_jersey_feature(frame, player)
            if feature is None:
                continue
            features.append(feature)
            feature_players.append(player)
            feature_map[player.id] = feature
            player.jersey_color = color

        if len(features) < 2:
            for player in players:
                player.team = "Team A"
            return

        data = np.array(features, dtype=np.float32)
        _, labels, centers = cv2.kmeans(
            data,
            2,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 18, 0.45),
            8,
            cv2.KMEANS_PP_CENTERS,
        )
        for player, label in zip(feature_players, labels.flatten(), strict=False):
            player.team = "Team A" if int(label) == 0 else "Team B"

        labeled_players = [player for player in feature_players if player.team is not None]
        for player in players:
            if player.team is not None:
                continue
            relaxed_feature, color = self._extract_jersey_feature(frame, player, relaxed=True)
            if relaxed_feature is not None:
                player.jersey_color = color
                distances = [float(np.linalg.norm(relaxed_feature - center)) for center in centers]
                player.team = "Team A" if int(np.argmin(distances)) == 0 else "Team B"
                continue
            if labeled_players:
                nearest = min(
                    labeled_players,
                    key=lambda candidate: math.hypot(
                        candidate.feet_point[0] - player.feet_point[0],
                        candidate.feet_point[1] - player.feet_point[1],
                    ),
                )
                player.team = nearest.team
            else:
                player.team = "Team A"

    def _nearest_player_to_ball(self, players: list[Player], ball_center: tuple[float, float]) -> Player:
        return min(players, key=lambda player: math.hypot(player.feet_point[0] - ball_center[0], player.feet_point[1] - ball_center[1]))

    def _resolve_offside_context(
        self,
        players: list[Player],
        ball: dict[str, Any] | None,
        future_balls: list[tuple[float, float]],
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> tuple[Player, Player, Player, str, str] | None:
        if ball is not None:
            passer = self._nearest_player_to_ball(players, ball["center"])
            attack_direction = self._infer_attack_direction(players, passer, future_balls, frame_shape[1], geometry)
            attackers = [player for player in players if player.team == passer.team]
            defenders = [player for player in players if player.team != passer.team]
            if len(attackers) >= 2 and len(defenders) >= 2:
                receiver = self._select_receiver(attackers, passer, attack_direction, future_balls, geometry, frame_shape)
                second_last_defender = self._select_second_last_defender(defenders, attack_direction, geometry, frame_shape)
                return passer, receiver, second_last_defender, attack_direction, "ball"

        attack_direction = self._infer_attack_direction_without_ball(players, geometry, frame_shape)
        team_split = self._infer_attacking_and_defending_teams(players, attack_direction, geometry, frame_shape)
        if team_split is None:
            return None

        attackers, defenders = team_split
        receiver = self._select_receiver_without_ball(attackers, defenders, attack_direction, geometry, frame_shape)
        passer = self._select_passer_without_ball(attackers, receiver, attack_direction, geometry, frame_shape)
        second_last_defender = self._select_second_last_defender(defenders, attack_direction, geometry, frame_shape)
        return passer, receiver, second_last_defender, attack_direction, "shape"

    def _track_future_ball_positions(self, video_path: Path, frame_timestamp: float) -> list[tuple[float, float]]:
        positions: list[tuple[float, float]] = []
        for delta in (0.12, 0.24, 0.36):
            try:
                frame, _ = extract_frame(video_path, frame_timestamp + delta)
                result = self._detect()(frame, device=self.device, conf=0.12, imgsz=960, verbose=False)[0]
                ball = self._extract_ball(result)
                if ball:
                    positions.append(ball["center"])
            except Exception:
                continue
        return positions

    def _infer_attack_direction(
        self,
        players: list[Player],
        passer: Player,
        future_balls: list[tuple[float, float]],
        frame_width: int,
        geometry: dict[str, Any],
    ) -> str:
        if future_balls:
            delta_x = future_balls[-1][0] - passer.feet_point[0]
            if abs(delta_x) >= 6:
                return "right" if delta_x > 0 else "left"
        if geometry["goal_line_right_x"] is not None and geometry["goal_line_left_x"] is None:
            return "right"
        if geometry["goal_line_left_x"] is not None and geometry["goal_line_right_x"] is None:
            return "left"
        sorted_players = sorted(players, key=lambda player: player.feet_point[0])
        if passer.feet_point[0] <= frame_width / 2:
            return "right" if passer != sorted_players[-1] else "left"
        return "left" if passer != sorted_players[0] else "right"

    def _infer_attack_direction_without_ball(
        self,
        players: list[Player],
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> str:
        if geometry["goal_line_right_x"] is not None and geometry["goal_line_left_x"] is None:
            return "right"
        if geometry["goal_line_left_x"] is not None and geometry["goal_line_right_x"] is None:
            return "left"
        width = frame_shape[1]
        left_edge = min(player.feet_point[0] for player in players)
        right_edge = max(player.feet_point[0] for player in players)
        return "right" if (width - right_edge) < left_edge else "left"

    def _infer_attacking_and_defending_teams(
        self,
        players: list[Player],
        attack_direction: str,
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> tuple[list[Player], list[Player]] | None:
        team_a = [player for player in players if player.team == "Team A"]
        team_b = [player for player in players if player.team == "Team B"]
        if len(team_a) < 2 or len(team_b) < 2:
            return None

        team_a_front = self._team_front_metric(team_a, attack_direction, geometry, frame_shape)
        team_b_front = self._team_front_metric(team_b, attack_direction, geometry, frame_shape)
        if attack_direction == "right":
            return (team_a, team_b) if team_a_front >= team_b_front else (team_b, team_a)
        return (team_a, team_b) if team_a_front <= team_b_front else (team_b, team_a)

    def _team_front_metric(
        self,
        players: list[Player],
        attack_direction: str,
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> float:
        metrics = sorted(self._projection_metric(player, geometry, frame_shape) for player in players)
        if attack_direction == "right":
            return mean(metrics[-min(2, len(metrics)) :])
        return mean(metrics[: min(2, len(metrics))])

    def _select_receiver(
        self,
        attackers: list[Player],
        passer: Player,
        attack_direction: str,
        future_balls: list[tuple[float, float]],
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> Player:
        candidates = [player for player in attackers if player.id != passer.id] or attackers
        if future_balls:
            target = future_balls[-1]
            return min(candidates, key=lambda player: math.hypot(player.feet_point[0] - target[0], player.feet_point[1] - target[1]))
        ordered = self._sort_players_by_attack_progress(candidates, attack_direction, geometry, frame_shape)
        return ordered[0]

    def _select_receiver_without_ball(
        self,
        attackers: list[Player],
        defenders: list[Player],
        attack_direction: str,
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> Player:
        ordered = self._sort_players_by_attack_progress(attackers, attack_direction, geometry, frame_shape)
        shortlist = ordered[: min(4, len(ordered))]
        if len(shortlist) == 1:
            return shortlist[0]

        metrics = [self._projection_metric(player, geometry, frame_shape) for player in attackers]
        metric_floor = min(metrics)
        metric_ceiling = max(metrics)
        metric_span = max(1.0, metric_ceiling - metric_floor)

        defender_lines = [self._projection_metric(player, geometry, frame_shape) for player in defenders]
        defender_anchor = float(mean(defender_lines)) if defender_lines else float(mean(metrics))
        action_y = float(np.median([player.feet_point[1] for player in defenders + shortlist]))
        width = frame_shape[1]
        height = frame_shape[0]

        def candidate_score(player: Player) -> float:
            progress_metric = self._projection_metric(player, geometry, frame_shape)
            if attack_direction == "right":
                progress = (progress_metric - metric_floor) / metric_span
            else:
                progress = (metric_ceiling - progress_metric) / metric_span

            nearest_defender = min(
                math.hypot(player.feet_point[0] - defender.feet_point[0], player.feet_point[1] - defender.feet_point[1])
                for defender in defenders
            )
            defender_proximity = 1.0 - min(1.0, nearest_defender / 360.0)
            line_proximity = 1.0 - min(1.0, abs(progress_metric - defender_anchor) / max(90.0, width * 0.18))
            lane_alignment = 1.0 - min(1.0, abs(player.feet_point[1] - action_y) / max(85.0, height * 0.22))

            x, y = player.feet_point
            edge_penalty = 0.0
            if x < width * 0.05 or x > width * 0.95:
                edge_penalty += 0.22
            if y < height * 0.08 or y > height * 0.95:
                edge_penalty += 0.18

            return progress * 0.38 + defender_proximity * 0.24 + line_proximity * 0.23 + lane_alignment * 0.19 - edge_penalty

        return max(shortlist, key=candidate_score)

    def _select_passer_without_ball(
        self,
        attackers: list[Player],
        receiver: Player,
        attack_direction: str,
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> Player:
        receiver_metric = self._projection_metric(receiver, geometry, frame_shape)
        if attack_direction == "right":
            candidates = [
                player for player in attackers if player.id != receiver.id and self._projection_metric(player, geometry, frame_shape) <= receiver_metric - 5
            ]
        else:
            candidates = [
                player for player in attackers if player.id != receiver.id and self._projection_metric(player, geometry, frame_shape) >= receiver_metric + 5
            ]
        if candidates:
            return min(candidates, key=lambda player: math.hypot(player.feet_point[0] - receiver.feet_point[0], player.feet_point[1] - receiver.feet_point[1]))
        ordered = self._sort_players_by_attack_progress([player for player in attackers if player.id != receiver.id], attack_direction, geometry, frame_shape)
        return ordered[0] if ordered else receiver

    def _select_second_last_defender(
        self,
        defenders: list[Player],
        attack_direction: str,
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> Player:
        reverse = attack_direction == "right"
        sorted_defenders = sorted(defenders, key=lambda player: self._projection_metric(player, geometry, frame_shape), reverse=reverse)
        return sorted_defenders[min(1, len(sorted_defenders) - 1)]

    def _resolve_manual_offside_context(
        self,
        players: list[Player],
        manual_selection: dict[str, Any],
    ) -> tuple[Player, Player, str] | None:
        attacker_id = str(manual_selection.get("attacker_id", "") or "")
        defender_id = str(manual_selection.get("defender_id", "") or "")
        attack_direction = str(manual_selection.get("attack_direction", "right"))
        attacker = self._find_player_by_id(players, attacker_id)
        defender = self._find_player_by_id(players, defender_id)
        if attacker is None:
            attacker = self._manual_point_player(manual_selection.get("attacker_point"), "manual_attacker", players)
        if defender is None:
            defender = self._manual_point_player(manual_selection.get("defender_point"), "manual_defender", players)
        if attacker is None or defender is None or attacker.id == defender.id:
            return None
        return attacker, defender, "left" if attack_direction == "left" else "right"

    def _manual_point_player(
        self,
        raw_point: Any,
        player_id: str,
        players: list[Player],
    ) -> Player | None:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
            return None
        try:
            x = float(raw_point[0])
            y = float(raw_point[1])
        except (TypeError, ValueError):
            return None
        widths = [player.bbox[2] - player.bbox[0] for player in players if player.bbox[2] > player.bbox[0]]
        heights = [player.bbox[3] - player.bbox[1] for player in players if player.bbox[3] > player.bbox[1]]
        width = float(np.median(widths)) if widths else 42.0
        height = float(np.median(heights)) if heights else 110.0
        x1 = x - width / 2
        x2 = x + width / 2
        y1 = y - height
        y2 = y
        return Player(
            id=player_id,
            bbox=(x1, y1, x2, y2),
            confidence=1.0,
            keypoints=None,
            team="Manual",
            jersey_color=None,
            pitch_score=1.0,
            on_pitch=True,
        )

    def _find_player_by_id(self, players: list[Player], player_id: str) -> Player | None:
        for player in players:
            if player.id == player_id:
                return player
        return None

    def _finalize_offside_result(
        self,
        *,
        frame: np.ndarray,
        output_dir: Path,
        actual_time: float,
        geometry: dict[str, Any],
        ball: dict[str, Any] | None,
        future_balls: list[tuple[float, float]],
        players: list[Player],
        attacker: Player,
        defender: Player,
        attack_direction: str,
        diagnostics: dict[str, Any],
        player_candidates: list[dict[str, Any]],
        locked_passer: Player | None,
        context_mode: str,
        selection_mode: str,
    ) -> dict[str, Any]:
        suggestions = [
            self._player_suggestion("attacker", attacker),
            self._player_suggestion("defender", defender),
        ]
        if locked_passer is not None:
            suggestions.insert(0, self._player_suggestion("passer", locked_passer))

        attacker_metric = self._projection_metric(attacker, geometry, frame.shape)
        defender_metric = self._projection_metric(defender, geometry, frame.shape)
        margin = max(4.0, frame.shape[1] * 0.0045)
        is_offside = attacker_metric > defender_metric + margin if attack_direction == "right" else attacker_metric < defender_metric - margin

        cluster_score = self._team_cluster_score(players)
        player_factor = min(1.0, len(players) / 9.0)
        goal_line_visible = geometry["goal_line_left_x"] is not None or geometry["goal_line_right_x"] is not None
        confidence = min(
            0.97,
            0.22
            + geometry["line_strength"] * 0.32
            + cluster_score * 0.24
            + player_factor * 0.16
            + (0.08 if goal_line_visible else 0.03)
            + (min(0.12, ball["confidence"] * 0.12) if ball else (0.05 if future_balls else 0.0)),
        )
        forced_call_ready = selection_mode == "manual" or confidence >= (0.48 if ball is not None else 0.46)
        if selection_mode == "manual":
            confidence = max(confidence, 0.84)

        if forced_call_ready:
            verdict = "Offside" if is_offside else "Onside"
            if selection_mode == "manual":
                rationale = (
                    "Manual correction applied. The selected attacker projects beyond the defending reference line at the locked pass frame."
                    if is_offside
                    else "Manual correction applied. The selected attacker stays level with or behind the defending reference line at the locked pass frame."
                )
            elif ball is not None:
                rationale = (
                    "The attacker projects beyond the defending reference line at the locked pass frame."
                    if is_offside
                    else "The attacker stays level with or behind the defending reference line at the locked pass frame."
                )
            else:
                rationale = (
                    "The player lines and pitch projection are strong enough to confirm the attacker is beyond the defending reference line at the locked pass frame."
                    if is_offside
                    else "The player lines and pitch projection are strong enough to confirm the attacker stays level with or behind the defending reference line."
                )
        else:
            verdict = "Human Review"
            rationale = "The system resolved candidate players, but the separation line still needs a cleaner frame before forcing a call."

        diagnostics.update(
            {
                "attack_direction": attack_direction,
                "cluster_score": round(cluster_score, 3),
                "attacker_metric": round(attacker_metric, 2),
                "defender_metric": round(defender_metric, 2),
                "context_mode": context_mode,
                "goal_line_visible": goal_line_visible,
                "passer_locked": locked_passer is not None,
                "selection_mode": selection_mode,
            }
        )
        snapshot_path = output_dir / "offside_snapshot.jpg"
        diagram_path = output_dir / "offside_diagram.png"
        annotated = self._draw_offside_overlay(
            frame,
            attacker,
            defender,
            locked_passer,
            ball,
            geometry,
            attack_direction,
            verdict,
        )
        cv2.imwrite(str(snapshot_path), annotated)
        self._draw_diagram(diagram_path, attacker, defender, attack_direction, verdict, geometry, frame.shape)
        return {
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "frame_timestamp": actual_time,
            "frame_source_path": output_dir / "offside_frame.jpg",
            "snapshot_path": snapshot_path,
            "diagram_path": diagram_path,
            "rationale": rationale,
            "diagnostics": diagnostics,
            "suggestions": suggestions,
            "player_candidates": player_candidates,
            "selected_attacker_id": attacker.id,
            "selected_defender_id": defender.id,
            "attack_direction": attack_direction,
        }

    def _projection_metric(self, player: Player, geometry: dict[str, Any], frame_shape: tuple[int, int, int]) -> float:
        x, y = player.feet_point
        height = frame_shape[0]
        slope = float(geometry["slope"])
        return x - slope * (height - y)

    def _sort_players_by_attack_progress(
        self,
        players: list[Player],
        attack_direction: str,
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> list[Player]:
        reverse = attack_direction == "right"
        return sorted(players, key=lambda player: self._projection_metric(player, geometry, frame_shape), reverse=reverse)

    def _team_cluster_score(self, players: list[Player]) -> float:
        colors = [player.jersey_color for player in players if player.jersey_color]
        if len(colors) < 2:
            return 0.18
        vectors = np.array(colors, dtype=np.float32)
        spread = np.linalg.norm(vectors.max(axis=0) - vectors.min(axis=0))
        return min(1.0, spread / 150.0)

    def _extract_jersey_feature(
        self,
        frame: np.ndarray,
        player: Player,
        relaxed: bool = False,
    ) -> tuple[np.ndarray | None, tuple[int, int, int] | None]:
        x1, y1, x2, y2 = player.bbox
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        inset = 0.18 if not relaxed else 0.1
        top = 0.12 if not relaxed else 0.08
        bottom = 0.58 if not relaxed else 0.68

        crop_x1 = int(max(0, x1 + width * inset))
        crop_x2 = int(min(frame.shape[1], x2 - width * inset))
        crop_y1 = int(max(0, y1 + height * top))
        crop_y2 = int(min(frame.shape[0], y1 + height * bottom))
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return None, None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        value_mask = hsv[..., 2] > (55 if relaxed else 70)
        vivid_mask = hsv[..., 1] > 38
        bright_neutral_mask = (hsv[..., 1] < 48) & (hsv[..., 2] > 150)
        non_pitch_hue = (hsv[..., 0] < 26) | (hsv[..., 0] > 98)
        candidate_mask = value_mask & ((vivid_mask & non_pitch_hue) | bright_neutral_mask)
        pixels = crop[candidate_mask]
        if len(pixels) < (18 if relaxed else 26):
            return None, None

        lab_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        feature = np.median(lab_pixels, axis=0).astype(np.float32)
        color = tuple(int(value) for value in np.median(pixels, axis=0))
        return feature, color

    def _player_suggestion(self, label: str, player: Player) -> dict[str, Any]:
        return {
            "id": player.id,
            "label": label,
            "team": player.team,
            "confidence": round(player.confidence, 2),
            "bbox": [round(value, 1) for value in player.bbox],
        }

    def _player_candidate(self, player: Player) -> dict[str, Any]:
        return {
            "id": player.id,
            "team": player.team,
            "confidence": round(player.confidence, 2),
            "bbox": [round(value, 1) for value in player.bbox],
            "feet_point": [round(player.feet_point[0], 1), round(player.feet_point[1], 1)],
            "pitch_score": round(player.pitch_score, 3),
            "on_pitch": bool(player.on_pitch),
        }

    def _line_pitch_ratio(self, pitch_region: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        samples = 10
        hits = 0
        height, width = pitch_region.shape[:2]
        for step in range(samples + 1):
            t = step / max(1, samples)
            x = int(round(x1 + (x2 - x1) * t))
            y = int(round(y1 + (y2 - y1) * t))
            x_start = max(0, x - 2)
            x_end = min(width, x + 3)
            y_start = max(0, y - 2)
            y_end = min(height, y + 3)
            patch = pitch_region[y_start:y_end, x_start:x_end]
            if patch.size and float(patch.mean()) >= 100.0:
                hits += 1
        return hits / (samples + 1)

    def _normalize_direction(self, dx: float, dy: float) -> tuple[float, float] | None:
        magnitude = math.hypot(dx, dy)
        if magnitude <= 1e-6:
            return None
        if dy < 0:
            dx *= -1
            dy *= -1
        return dx / magnitude, dy / magnitude

    def _bbox_iou(self, a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter_area / (area_a + area_b - inter_area)

    def _draw_offside_overlay(
        self,
        frame: np.ndarray,
        attacker: Player,
        defender: Player,
        passer: Player | None,
        ball: dict[str, Any] | None,
        geometry: dict[str, Any],
        attack_direction: str,
        verdict: str,
    ) -> np.ndarray:
        annotated = frame.copy()
        colors = {
            "attacker": (45, 177, 255),
            "defender": (76, 220, 132),
            "passer": (240, 180, 55),
        }
        players_to_draw: list[tuple[Player, tuple[int, int, int]]] = [
            (attacker, colors["attacker"]),
            (defender, colors["defender"]),
        ]
        if passer is not None:
            players_to_draw.append((passer, colors["passer"]))

        for player, color in players_to_draw:
            x1, y1, x2, y2 = map(int, player.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            foot_x, foot_y = map(int, player.feet_point)
            cv2.circle(annotated, (foot_x, foot_y), 5, color, -1)
            cv2.circle(annotated, (foot_x, foot_y), 10, color, 1)

        if ball is not None:
            center_x, center_y = map(int, ball["center"])
            cv2.circle(annotated, (center_x, center_y), max(6, int(ball["radius"])), (255, 255, 255), 2)
            cv2.circle(annotated, (center_x, center_y), 3, (255, 255, 255), -1)

        defender_x, defender_y = defender.feet_point
        width = annotated.shape[1]
        height = annotated.shape[0]
        local_direction = self._select_local_line_direction((defender_x, defender_y), geometry)
        vanishing_direction = self._vanishing_direction((defender_x, defender_y), geometry)
        line_direction = self._resolve_overlay_line_direction(local_direction, vanishing_direction, geometry.get("line_direction"))
        if line_direction is not None:
            start, end = self._extend_line_through_point((defender_x, defender_y), line_direction, width, height)
        else:
            x_position = int(defender_x)
            start, end = (x_position, 0), (x_position, height - 1)
        cv2.line(annotated, start, end, (43, 56, 215) if verdict == "Offside" else (94, 235, 104), 3)
        cv2.putText(
            annotated,
            f"{verdict.upper()} | {attack_direction.upper()} ATTACK",
            (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (242, 247, 252),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _extend_line_through_point(
        self,
        point: tuple[float, float],
        direction: tuple[float, float],
        width: int,
        height: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        px, py = point
        dx, dy = direction
        intersections: list[tuple[float, tuple[int, int]]] = []

        if abs(dx) > 1e-6:
            for edge_x in (0.0, float(width - 1)):
                t = (edge_x - px) / dx
                y = py + t * dy
                if 0.0 <= y <= height - 1:
                    intersections.append((t, (int(round(edge_x)), int(round(y)))))
        if abs(dy) > 1e-6:
            for edge_y in (0.0, float(height - 1)):
                t = (edge_y - py) / dy
                x = px + t * dx
                if 0.0 <= x <= width - 1:
                    intersections.append((t, (int(round(x)), int(round(edge_y)))))

        if len(intersections) < 2:
            x_position = int(round(px))
            return (x_position, 0), (x_position, height - 1)

        intersections.sort(key=lambda item: item[0])
        return intersections[0][1], intersections[-1][1]

    def _vanishing_direction(
        self,
        anchor: tuple[float, float],
        geometry: dict[str, Any],
    ) -> tuple[float, float] | None:
        vanishing_point = geometry.get("vanishing_point")
        if vanishing_point is None:
            return None
        vx, vy = vanishing_point
        ax, ay = anchor
        return self._normalize_direction(vx - ax, vy - ay)

    def _resolve_overlay_line_direction(
        self,
        local_direction: tuple[float, float] | None,
        vanishing_direction: tuple[float, float] | None,
        global_direction: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if local_direction is None:
            return vanishing_direction or global_direction
        if vanishing_direction is None:
            return local_direction

        local_dx = abs(local_direction[0])
        vanishing_dx = abs(vanishing_direction[0])

        if vanishing_dx < max(0.08, local_dx * 0.58):
            return local_direction

        blended = self._normalize_direction(
            local_direction[0] * 0.72 + vanishing_direction[0] * 0.28,
            local_direction[1] * 0.72 + vanishing_direction[1] * 0.28,
        )
        return blended or local_direction or vanishing_direction or global_direction

    def _select_local_line_direction(
        self,
        anchor: tuple[float, float],
        geometry: dict[str, Any],
    ) -> tuple[float, float] | None:
        candidates = geometry.get("candidate_lines") or []
        if not candidates:
            return None
        anchor_x, anchor_y = anchor
        best_direction: tuple[float, float] | None = None
        best_score = -1e9
        for score, _, _, line, source in candidates:
            x1, y1, x2, y2 = line
            direction = self._normalize_direction(float(x2 - x1), float(y2 - y1))
            if direction is None:
                continue
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            distance_penalty = math.hypot(anchor_x - mid_x, anchor_y - mid_y) / 360.0
            source_bias = 1.0 if source == "white" else 0.82
            candidate_score = score * source_bias - distance_penalty * score * 0.52
            if candidate_score > best_score:
                best_score = candidate_score
                best_direction = direction
        return best_direction

    def _goal_line_clearance(self, ball: dict[str, Any], line_x: float, goal_side: str) -> float:
        ball_x, _ = ball["center"]
        radius = float(ball["radius"])
        if goal_side == "right":
            return (ball_x - radius) - line_x
        return line_x - (ball_x + radius)

    def _goal_clearance_margin(self, ball: dict[str, Any], frame_shape: tuple[int, int, int]) -> float:
        radius = float(ball["radius"])
        return max(1.5, min(4.0, radius * 0.08 + frame_shape[1] * 0.0012))

    def _goal_boundary_state(self, clearance_px: float, clear_margin_px: float) -> str:
        if clearance_px > clear_margin_px:
            return "clear"
        if clearance_px >= -clear_margin_px:
            return "touching"
        return "overlap"

    def _draw_goal_overlay(
        self,
        frame: np.ndarray,
        ball: dict[str, Any],
        line_x: float | None,
        goal_side: str,
        verdict: str,
        boundary_state: str,
        clearance_px: float,
        clear_margin_px: float,
    ) -> np.ndarray:
        annotated = frame.copy()
        center_x, center_y = map(int, ball["center"])
        radius = max(6, int(ball["radius"]))
        line_color = (82, 210, 122) if verdict == "Goal" else (255, 205, 82) if verdict == "No Goal" else (94, 206, 255)
        cv2.circle(annotated, (center_x, center_y), radius, (255, 255, 255), 2)
        cv2.circle(annotated, (center_x, center_y), 3, (255, 255, 255), -1)
        if line_x is not None:
            cv2.line(annotated, (int(line_x), 0), (int(line_x), annotated.shape[0]), line_color, 3)
        status_copy = {
            "clear": f"CLEAR GAP {max(0.0, clearance_px):.1f}px",
            "touching": "TOUCHING LINE",
            "overlap": f"LINE OVERLAP {abs(min(0.0, clearance_px)):.1f}px",
        }.get(boundary_state, "GOAL LINE REVIEW")
        cv2.putText(
            annotated,
            f"{verdict.upper()} | {goal_side.upper()} GOAL",
            (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (242, 247, 252),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"{status_copy}  |  threshold {clear_margin_px:.1f}px",
            (24, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (232, 239, 245),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _draw_goal_diagram(
        self,
        output_path: Path,
        goal_side: str,
        verdict: str,
        boundary_state: str,
        clearance_px: float,
        clear_margin_px: float,
    ) -> None:
        width, height = 960, 560
        image = Image.new("RGBA", (width, height), "#081a2d")
        draw = ImageDraw.Draw(image)
        verdict_color = "#52d27d" if verdict == "Goal" else "#f3b840" if verdict == "No Goal" else "#7cc7ff"
        line_color = "#6fe1ff" if verdict == "Human Review" else verdict_color
        ball_fill = "#eef5fb"
        ball_outline = verdict_color
        panel = (40, 40, width - 40, height - 40)
        draw.rounded_rectangle(panel, radius=28, outline=(120, 176, 208, 34), width=1, fill=(8, 24, 39, 84))
        draw.rounded_rectangle((88, 118, width - 88, height - 104), radius=24, outline=(110, 197, 240, 36), width=1, fill=(10, 31, 48, 168))

        line_x = width // 2
        top_y, bottom_y = 150, height - 140
        draw.line([(line_x, top_y), (line_x, bottom_y)], fill=line_color, width=10)
        draw.line([(line_x, top_y), (line_x, bottom_y)], fill=(255, 255, 255, 36), width=2)

        if boundary_state == "clear":
            display_gap = max(18.0, min(120.0, clearance_px * 8.0))
        elif boundary_state == "touching":
            display_gap = 0.0
        else:
            display_gap = -max(12.0, min(72.0, abs(clearance_px) * 0.35))
        radius = 72
        if goal_side == "left":
            ball_center_x = line_x - radius - display_gap
        else:
            ball_center_x = line_x + radius + display_gap
        ball_center_y = (top_y + bottom_y) // 2
        draw.ellipse(
            (ball_center_x - radius, ball_center_y - radius, ball_center_x + radius, ball_center_y + radius),
            fill=ball_fill,
            outline=ball_outline,
            width=8,
        )
        draw.ellipse(
            (ball_center_x - radius + 16, ball_center_y - radius + 16, ball_center_x + radius - 16, ball_center_y + radius - 16),
            outline=(16, 40, 60, 52),
            width=2,
        )

        if boundary_state == "clear":
            gap_start = line_x if goal_side == "right" else ball_center_x + radius
            gap_end = ball_center_x - radius if goal_side == "right" else line_x
            draw.line([(gap_start, ball_center_y - 110), (gap_end, ball_center_y - 110)], fill="#ffffff", width=4)
            draw.line([(gap_start, ball_center_y - 120), (gap_start, ball_center_y - 100)], fill="#ffffff", width=4)
            draw.line([(gap_end, ball_center_y - 120), (gap_end, ball_center_y - 100)], fill="#ffffff", width=4)
        elif boundary_state == "touching":
            draw.line([(line_x, ball_center_y - radius - 18), (line_x, ball_center_y + radius + 18)], fill=(255, 255, 255, 72), width=2)

        try:
            title_font = ImageFont.truetype("arial.ttf", 40)
            body_font = ImageFont.truetype("arial.ttf", 26)
            meta_font = ImageFont.truetype("arial.ttf", 18)
        except OSError:
            title_font = body_font = meta_font = ImageFont.load_default()

        state_copy = {
            "clear": "Whole ball clear of the line",
            "touching": "Ball still touching the line",
            "overlap": "Ball still overlapping the line",
        }.get(boundary_state, "Goal line review")
        clearance_copy = (
            f"Visible gap: {max(0.0, clearance_px):.1f}px"
            if boundary_state == "clear"
            else f"Line contact margin: {abs(clearance_px):.1f}px"
        )

        draw.text((88, 62), "Goal-Line Diagram", font=title_font, fill="#e8f3ff")
        draw.text((88, 92), f"{verdict} · {goal_side.title()} Goal", font=body_font, fill=verdict_color)
        draw.text((88, height - 84), state_copy, font=body_font, fill="#edf4fb")
        draw.text((88, height - 52), f"{clearance_copy} · threshold {clear_margin_px:.1f}px", font=meta_font, fill="#9fb4c9")

        image.convert("RGB").save(output_path, format="PNG")

    def _draw_goal_diagram(
        self,
        output_path: Path,
        goal_side: str,
        verdict: str,
        boundary_state: str,
        clearance_px: float,
        clear_margin_px: float,
    ) -> None:
        width, height = 960, 560
        image = Image.new("RGBA", (width, height), "#081a2d")
        draw = ImageDraw.Draw(image)
        verdict_color = "#52d27d" if verdict == "Goal" else "#f3b840" if verdict == "No Goal" else "#7cc7ff"
        line_color = "#6fe1ff" if verdict == "Human Review" else verdict_color
        panel = (40, 40, width - 40, height - 40)
        draw.rounded_rectangle(panel, radius=28, outline=(120, 176, 208, 34), width=1, fill=(8, 24, 39, 84))
        draw.rounded_rectangle((88, 136, width - 88, height - 110), radius=24, outline=(110, 197, 240, 36), width=1, fill=(10, 31, 48, 168))

        line_x = width // 2
        top_y, bottom_y = 176, height - 156
        draw.line([(line_x, top_y), (line_x, bottom_y)], fill=line_color, width=10)
        draw.line([(line_x, top_y), (line_x, bottom_y)], fill=(255, 255, 255, 36), width=2)

        if boundary_state == "clear":
            display_gap = max(18.0, min(120.0, clearance_px * 8.0))
        elif boundary_state == "touching":
            display_gap = 0.0
        else:
            display_gap = -max(12.0, min(72.0, abs(clearance_px) * 0.35))

        radius = 72
        if goal_side == "left":
            ball_center_x = line_x - radius - display_gap
        else:
            ball_center_x = line_x + radius + display_gap
        ball_center_y = (top_y + bottom_y) // 2
        draw.ellipse(
            (ball_center_x - radius, ball_center_y - radius, ball_center_x + radius, ball_center_y + radius),
            fill="#eef5fb",
            outline=verdict_color,
            width=8,
        )
        draw.ellipse(
            (ball_center_x - radius + 16, ball_center_y - radius + 16, ball_center_x + radius - 16, ball_center_y + radius - 16),
            outline=(16, 40, 60, 52),
            width=2,
        )

        if boundary_state == "clear":
            gap_start = line_x if goal_side == "right" else ball_center_x + radius
            gap_end = ball_center_x - radius if goal_side == "right" else line_x
            draw.line([(gap_start, ball_center_y - 110), (gap_end, ball_center_y - 110)], fill="#ffffff", width=4)
            draw.line([(gap_start, ball_center_y - 120), (gap_start, ball_center_y - 100)], fill="#ffffff", width=4)
            draw.line([(gap_end, ball_center_y - 120), (gap_end, ball_center_y - 100)], fill="#ffffff", width=4)
        elif boundary_state == "touching":
            draw.line([(line_x, ball_center_y - radius - 18), (line_x, ball_center_y + radius + 18)], fill=(255, 255, 255, 72), width=2)

        try:
            title_font = ImageFont.truetype("arial.ttf", 38)
            body_font = ImageFont.truetype("arial.ttf", 24)
            meta_font = ImageFont.truetype("arial.ttf", 18)
        except OSError:
            title_font = body_font = meta_font = ImageFont.load_default()

        state_copy = {
            "clear": "Whole ball clear of the line",
            "touching": "Ball still touching the line",
            "overlap": "Ball still overlapping the line",
        }.get(boundary_state, "Goal line review")
        clearance_copy = (
            f"Visible gap: {max(0.0, clearance_px):.1f}px"
            if boundary_state == "clear"
            else f"Line contact margin: {abs(clearance_px):.1f}px"
        )

        draw.text((88, 56), "Goal-Line Diagram", font=title_font, fill="#e8f3ff")
        draw.text((88, 98), f"{verdict} | {goal_side.title()} Goal", font=body_font, fill=verdict_color)
        draw.text((88, height - 112), state_copy, font=body_font, fill="#edf4fb")
        draw.text((88, height - 78), f"{clearance_copy} | threshold {clear_margin_px:.1f}px", font=meta_font, fill="#9fb4c9")

        image.convert("RGB").save(output_path, format="PNG")

    def _draw_diagram(
        self,
        output_path: Path,
        attacker: Player,
        defender: Player,
        attack_direction: str,
        verdict: str,
        geometry: dict[str, Any],
        frame_shape: tuple[int, int, int],
    ) -> None:
        width, height = 960, 560
        image = Image.new("RGBA", (width, height), "#081a2d")
        draw = ImageDraw.Draw(image)
        accent = "#ff5a52" if verdict == "Offside" else "#52d27d" if verdict == "Onside" else "#9aa8b7"
        panel = [(152, 494), (808, 494), (732, 110), (228, 110)]
        draw.rounded_rectangle((38, 36, width - 38, height - 36), radius=26, outline=(120, 176, 208, 34), width=1, fill=(8, 24, 39, 72))
        draw.polygon(panel, fill="#102b3b", outline="#78ddb0")
        draw.polygon([(228, 110), (732, 110), (808, 494), (152, 494)], outline=(210, 249, 236, 24))
        for lane in range(5):
            start_x = 248 + lane * 106
            draw.line([(start_x, 494), (start_x + 72, 110)], fill="#175b79", width=3)

        attacker_metric = self._projection_metric(attacker, geometry, frame_shape)
        defender_metric = self._projection_metric(defender, geometry, frame_shape)
        attack_sign = 1 if attack_direction == "right" else -1
        projected_gap = (attacker_metric - defender_metric) * attack_sign
        scaled_gap = projected_gap * 0.22
        if abs(projected_gap) > 18:
            scaled_gap = math.copysign(max(54.0, abs(scaled_gap)), scaled_gap)
        scaled_gap = max(-220.0, min(220.0, scaled_gap))

        line_x = width // 2
        ground_y = 424
        attacker_x = int(round(line_x + attack_sign * scaled_gap))
        defender_x = line_x

        zone_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        zone_draw = ImageDraw.Draw(zone_overlay)
        if verdict == "Offside":
            zone_polygon = (
                [(line_x, 110), (732, 110), (808, 494), (line_x, 494)]
                if attack_direction == "right"
                else [(228, 110), (line_x, 110), (line_x, 494), (152, 494)]
            )
            zone_draw.polygon(zone_polygon, fill=(255, 90, 82, 28))
        image = Image.alpha_composite(image, zone_overlay)
        draw = ImageDraw.Draw(image)

        draw.line([(line_x, 110), (line_x, 494)], fill=accent, width=6)
        draw.line([(line_x, 110), (line_x, 494)], fill=(255, 255, 255, 42), width=1)
        draw.ellipse((line_x - 16, ground_y + 20, line_x + 16, ground_y + 36), fill="#06131f")
        draw.ellipse((attacker_x - 16, ground_y + 20, attacker_x + 16, ground_y + 36), fill="#06131f")

        self._draw_diagram_player(draw, defender_x, ground_y, "#5fdc88", "#effaf3")
        self._draw_diagram_player(draw, attacker_x, ground_y, "#45b1ff", "#eef7ff")

        bracket_y = 138
        marker_left = min(line_x, attacker_x)
        marker_right = max(line_x, attacker_x)
        if abs(marker_right - marker_left) >= 14:
            draw.line([(marker_left, bracket_y), (marker_right, bracket_y)], fill=accent, width=4)
            draw.line([(marker_left, bracket_y - 10), (marker_left, bracket_y + 10)], fill=accent, width=4)
            draw.line([(marker_right, bracket_y - 10), (marker_right, bracket_y + 10)], fill=accent, width=4)

        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        draw.text((74, 42), f"{verdict} focus", fill="#f5f7fb", font=title_font)
        draw.text((74, 74), "Attacker against defender line", fill="#8da5b8", font=body_font)
        draw.text((74, 100), f"Projected gap {abs(projected_gap):.0f}px", fill=accent, font=body_font)
        draw.text((line_x - 28, 84), "LINE", fill="#f4f7fb", font=body_font)
        draw.text((defender_x - 28, ground_y + 64), "Def", fill="#f5f7fb", font=body_font)
        draw.text((attacker_x - 24, ground_y + 64), "Att", fill="#f5f7fb", font=body_font)
        image.save(output_path)

    def _draw_diagram_player(
        self,
        draw: ImageDraw.ImageDraw,
        foot_x: int,
        ground_y: int,
        outline: str,
        fill: str,
    ) -> None:
        shoulder_y = ground_y - 146
        chest_bottom = ground_y - 92
        hip_y = ground_y - 58
        head_center_y = ground_y - 190
        shadow = (foot_x - 42, ground_y + 12, foot_x + 42, ground_y + 32)
        draw.ellipse(shadow, fill="#06131f")
        draw.ellipse((foot_x - 25, head_center_y - 25, foot_x + 25, head_center_y + 25), fill=fill, outline=outline, width=4)
        draw.rounded_rectangle((foot_x - 29, shoulder_y, foot_x + 29, shoulder_y + 20), radius=12, fill=fill, outline=outline, width=3)
        draw.rounded_rectangle((foot_x - 21, shoulder_y + 18, foot_x + 21, chest_bottom), radius=12, fill=fill, outline=outline, width=3)
        draw.rounded_rectangle((foot_x - 18, chest_bottom - 2, foot_x + 18, hip_y), radius=10, fill=fill, outline=outline, width=3)
        draw.line([(foot_x - 16, shoulder_y + 20), (foot_x - 30, chest_bottom + 6)], fill=outline, width=7)
        draw.line([(foot_x + 16, shoulder_y + 20), (foot_x + 30, chest_bottom + 6)], fill=outline, width=7)
        draw.line([(foot_x - 8, hip_y), (foot_x - 18, ground_y)], fill=outline, width=7)
        draw.line([(foot_x + 8, hip_y), (foot_x + 18, ground_y)], fill=outline, width=7)
        draw.line([(foot_x, head_center_y + 24), (foot_x, shoulder_y)], fill=outline, width=4)


_ANALYZER: VisionAnalyzer | None = None


def get_analyzer(settings: Settings) -> VisionAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = VisionAnalyzer(settings)
    return _ANALYZER
