# VAR Football

A prototype football incident review platform for single-camera offside and goal-line checks.

This project was built as a technical demo around a simple operator flow:

1. Create a match
2. Load a sample clip or upload a short video
3. Trigger an offside or goal-line review from the source timeline
4. Scrub the generated review window and lock the decision frame
5. Generate an annotated result and store it in the incident log

## Stack

### Frontend
- React
- TypeScript
- Vite
- Tailwind CSS
- Framer Motion

### Backend
- FastAPI
- Python
- OpenCV
- Ultralytics YOLO

## How the analysis works

### Offside check
- The source clip is trimmed into a short review window.
- The operator scrubs to the pass frame and locks it.
- OpenCV extracts that frame and handles image overlays, clip/frame I/O, and pitch-line processing.
- YOLO detection is used to find people and the ball.
- YOLO pose is used locally to estimate body points when available.
- The backend separates likely teams using jersey-color heuristics and estimates the relevant attacker and defender.
- Pitch markings and scene geometry are used to project an offside reference line onto the frame.
- If the automatic selection is weak, the operator can manually correct the attacker and defender.

### Goal-line check
- A short clip near the review timestamp is sampled.
- YOLO detection is used to locate the ball in candidate frames.
- OpenCV is used to estimate the goal-line position from the post/line geometry.
- The backend evaluates whether the whole ball is clearly beyond the line, touching it, or still overlapping it.
- A simplified goal-line diagram is generated alongside the annotated snapshot.

## Project structure

```text
frontend/   React client
backend/    FastAPI service and review pipeline
assets/     Bundled sample clips for the demo
```

## Local development

### Backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

## Environment variables

### Frontend

```text
VITE_API_BASE_URL=http://localhost:8000
```

### Backend

```text
APP_ENV=local
STORAGE_DIR=backend/storage
MODEL_DEVICE=auto
CORS_ORIGINS=http://localhost:5173
```

## Notes

- The deployed demo is best suited to bundled sample clips.
- Local development supports the fuller workflow, including uploads and manual correction.
- On small hosted instances, model inference can require a lighter deployment setup than local development.
