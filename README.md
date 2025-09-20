# hip-angle
mediapipe modelis, time-series data export
Hip Abduction – Mobile Pose (Web): A phone-friendly browser app that estimates per-hip abduction angles during pediatric hip casting using MediaPipe Pose; it overlays hips/knees, a pelvis midline, and angle arcs, 
computes left/right and average abduction with confidence-aware handling (landmark visibility), EMA smoothing, and an outlier guard to reduce jitter, then color-codes results (green 30–45°, amber 45–55°, red >55° or <30°) 
with on-screen warnings; includes a 3-second capture where only the 1–2 s interval is saved, exporting a JSON time series (timestamps, L/R/avg angles, pelvis vector, landmark coords + visibility); 
runs fully client-side (privacy-friendly) and can be hosted on GitHub Pages—just open the URL, allow camera, press Pradėti duomenų rinkimą, and Atsisiųsti JSON to download; research/education tool only, not a medical device.
