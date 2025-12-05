
import cv2
from typing import Dict

def visualize_conflicts(video_path: str, results: Dict, out_path: str = "annotated_conflicts.mp4") -> str:
    cap = cv2.VideoCapture(video_path)
    fps = results["fps"]
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    trajectories = results["trajectories"]
    conflicts = results["conflict_analysis"]["conflicts"]

    # (track1, track2) pairs that are ever in conflict
    conflict_pairs = {(c["track_1"], c["track_2"]) for c in conflicts}
    conflict_pairs |= {(b, a) for (a, b) in conflict_pairs}

    # frame -> list of (tid, x1, y1, x2, y2, cls, conf)
    frame_map = {}
    for tid, traj in trajectories.items():
        for (tid, f, x1, y1, x2, y2, cls, conf) in traj:
            frame_map.setdefault(f, []).append((tid, x1, y1, x2, y2, cls, conf))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        objs = frame_map.get(frame_idx, [])
        tids_in_frame = [o[0] for o in objs]
        tids_set = set(tids_in_frame)

        for tid, x1, y1, x2, y2, cls, conf in objs:
            # check if tid conflicts with any other tid present in THIS frame
            is_conflict = any((tid, other) in conflict_pairs for other in tids_set if other != tid)

            color = (0, 255, 0)  # green
            if is_conflict:
                color = (0, 0, 255)  # red

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label = f"ID {tid} C{int(cls)} {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved annotated video to {out_path}")
    return out_path
