
import cv2
import numpy as np

video_path = "/content/nnds_verify/sample_data/traffic_video.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}, total frames: {total_frames}, duration: {total_frames/fps:.1f}s")

# Grab a few frames spread through the video
frame_indices = [int(total_frames * f) for f in [0.1, 0.3, 0.5, 0.7, 0.9]]

for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {idx}")
        continue

    # Draw pixel grid every 100px with labels
    h, w = frame.shape[:2]
    for x in range(0, w, 100):
        cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(frame, str(x), (x+2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for y in range(0, h, 100):
        cv2.line(frame, (0, y), (w, y), (0, 255, 0), 1)
        cv2.putText(frame, str(y), (2, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    out_path = f"/content/nnds_verify/frame_{idx}_grid.jpg"
    cv2.imwrite(out_path, frame)
    print(f"Saved {out_path}")

cap.release()
