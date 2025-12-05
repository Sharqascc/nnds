
import cv2
import json
from typing import Dict
from datetime import datetime

from batch_inference import BatchedYOLOInference
from improved_tracking import TrajectoryTracker
from pet_ttc_calculator import analyze_conflicts


class TrafficAnalysisPipeline:
    def __init__(self, model_path: str, output_dir: str = './results', fps: float = 30.0):
        self.inference = BatchedYOLOInference(model_path, batch_size=8, conf_threshold=0.05)
        self.tracker = TrajectoryTracker(iou_threshold=0.3, max_age=5)
        self.output_dir = output_dir
        self.fps = fps
        self.pet_threshold = 3.0

    def process_video(self, video_path: str, sample_rate: int = 1) -> Dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frame_idx = 0
        frames_to_process = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                frames_to_process.append((frame_idx, frame))
            frame_idx += 1

        cap.release()
        total_frames = frame_idx

        detections = self.inference.process_frames(frames_to_process)

        for det in detections:
            self.tracker.update(det, det['frame_idx'])

        tracking_stats = self.tracker.get_statistics()
        trajectories_dict = self.tracker.get_track_trajectories()
        conflict_analysis = analyze_conflicts(
            trajectories_dict,
            fps=self.fps,
            pet_threshold=self.pet_threshold
        )

        results = {
            "timestamp": datetime.now().isoformat(),
            "video_path": video_path,
            "total_frames": total_frames,
            "frames_processed": len(frames_to_process),
            "fps": self.fps,
            "inference_stats": self.inference.get_model_info(),
            "tracking_stats": tracking_stats,
            "conflict_analysis": conflict_analysis
        }
        return results

    def save_results(self, results: Dict, filename: str = "analysis_results.json") -> str:
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = f"{self.output_dir}/{filename}"
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")
        return filepath

    def generate_report(self, results: Dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("TRAFFIC SAFETY ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"Video: {results['video_path']}")
        lines.append(f"Frames: {results['frames_processed']} / {results['total_frames']}")
        lines.append(f"FPS: {results['fps']}")
        lines.append("")
        lines.append("TRACKING")
        lines.append("-" * 60)
        ts = results['tracking_stats']
        lines.append(f"Total tracks: {ts['total_tracks']}")
        lines.append(f"Active tracks: {ts['active_tracks']}")
        lines.append(f"Trajectory points: {ts['trajectory_points']}")
        lines.append("")
        lines.append("CONFLICTS (PET in seconds)")
        lines.append("-" * 60)
        ca = results['conflict_analysis']
        lines.append(f"Conflicts: {ca['num_conflicts']}")
        lines.append(f"Tracks analyzed: {ca['tracks_analyzed']}")
        for c in sorted(ca['conflicts'], key=lambda x: x['pet_seconds']):
            lines.append(f"  {c['track_1']} vs {c['track_2']}: PET={c['pet_seconds']:.2f}s ({c['severity']})")
        lines.append("=" * 60)
        return "\n".join(lines)

    def save_report(self, results: Dict, filename: str = "report.txt") -> str:
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        text = self.generate_report(results)
        path = f"{self.output_dir}/{filename}"
        with open(path, "w") as f:
            f.write(text)
        print(f"Report saved to {path}")
        return path
