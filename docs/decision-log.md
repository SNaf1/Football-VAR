# Atlético Intelligence Decision Log

- The first working version is local-first and CPU-first.
- Video trimming and frame extraction use OpenCV so the app works without a local `ffmpeg` install.
- Offside review uses a locked pass frame plus a tiny nearby-frame window for ball context.
- Goal review samples a small frame set instead of scanning every frame in the clip.
- The faux-3D explainer card is intentionally stylized rather than true 3D reconstruction.
