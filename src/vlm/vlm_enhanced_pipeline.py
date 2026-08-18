import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from vlm.analyzer import VLLMAnalyzer
from tqdm import tqdm

class VLMEnhancedPipeline:
    def __init__(self, yolo_model_path=None, vlm_model_name="Salesforce/blip2-opt-2.7b"):
        self.yolo = YOLO(yolo_model_path) if yolo_model_path and os.path.exists(yolo_model_path) else None
        self.vlm = VLLMAnalyzer(model_name=vlm_model_name)
        self.results = []

    def process_video(self, video_path, output_dir="outputs/vlm_advanced_analysis", max_frames=30):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Processing video: {video_path} ({frame_count} frames, {fps:.2f} fps)")

        frames_processed = 0
        frame_interval = max(1, frame_count // max_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frames_processed % frame_interval == 0 and frames_processed < max_frames * frame_interval:
                img_path = os.path.join(output_dir, f"frame_{frames_processed:06d}.jpg")
                cv2.imwrite(img_path, frame)
                
                # Simulate trajectory data (since we don't have real tracking here)
                if self.yolo:
                    detections = self.yolo(frame)
                    boxes = []
                    for r in detections:
                        if r.boxes is not None:
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                boxes.append({
                                    'track_id': len(boxes)+1,
                                    'class': self.yolo.names[cls],
                                    'x': (x1+x2)/2,
                                    'y': (y1+y2)/2,
                                    'speed': np.random.uniform(10, 50),
                                    'pet': np.random.uniform(1.0, 5.0)
                                })
                    traj_df = pd.DataFrame(boxes)
                else:
                    # Dummy trajectory
                    dummy_data = {
                        'track_id': [1, 2],
                        'class': ['car', 'truck'],
                        'x': [100, 200],
                        'y': [150, 250],
                        'speed': [30, 40],
                        'pet': [1.5, 2.0]
                    }
                    traj_df = pd.DataFrame(dummy_data)

                vlm_result = self.vlm.analyze_with_trajectory(img_path, traj_df)
                self.results.append({
                    'frame': frames_processed,
                    'image': img_path,
                    'vlm_output': vlm_result,
                    'detections': boxes if self.yolo else []
                })
                
                frame_json_path = os.path.join(output_dir, f"frame_{frames_processed:06d}_analysis.json")
                with open(frame_json_path, 'w') as f:
                    json.dump(vlm_result, f, indent=2)

            frames_processed += 1
            if frames_processed >= max_frames * frame_interval:
                break

        cap.release()

        summary = {
            'total_frames_analyzed': len(self.results),
            'video_path': video_path,
            'timestamp': pd.Timestamp.now().isoformat(),
            'aggregated_risk': self._aggregate_risks()
        }
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Processing complete. Results saved to {output_dir}")
        return summary

    def _aggregate_risks(self):
        risks = []
        for r in self.results:
            risk = r['vlm_output'].get('risk_level', 'unknown')
            risks.append(risk)
        from collections import Counter
        return dict(Counter(risks))

    def generate_markdown_report(self, output_dir="outputs/vlm_advanced_analysis"):
        md_path = os.path.join(output_dir, 'report.md')
        with open(md_path, 'w') as f:
            f.write("# Enhanced VLM Safety Analysis Report\n\n")
            f.write(f"**Timestamp:** {pd.Timestamp.now().isoformat()}\n\n")
            f.write(f"**Frames Analyzed:** {len(self.results)}\n\n")
            f.write("## Risk Distribution\n")
            agg = self._aggregate_risks()
            for risk, count in agg.items():
                f.write(f"- {risk}: {count} frames\n")
            f.write("\n## Sample Frame Analyses\n")
            for idx, res in enumerate(self.results[:5]):
                f.write(f"### Frame {res['frame']}\n")
                f.write(f"**Image:** {res['image']}\n")
                f.write("**VLM Output:**\n")
                f.write(f"{json.dumps(res['vlm_output'], indent=2)}\n\n")
        print(f"✅ Markdown report saved to {md_path}")

if __name__ == "__main__":
    # Look for YOLO weights
    yolo_path = None
    for candidate in ["weights/best.pt", "yolo11n.pt", "yolo26n.pt"]:
        if os.path.exists(candidate):
            yolo_path = candidate
            break
    if yolo_path:
        print(f"Using YOLO model: {yolo_path}")
    else:
        print("No YOLO model found; running VLM-only mode.")

    pipeline = VLMEnhancedPipeline(yolo_model_path=yolo_path)

    video = "sample_data/traffic_video.mp4"
    if os.path.exists(video):
        summary = pipeline.process_video(video, max_frames=30)
        pipeline.generate_markdown_report()
    else:
        print("Video not found. Processing sample images from uvh26_data if available.")
        img_dir = "uvh26_data/UVH-26-Train/data/000/"
        if os.path.exists(img_dir):
            images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')][:5]
            for img in images:
                res = pipeline.vlm.analyze_with_trajectory(img, pd.DataFrame())
                print(f"Image: {img}\n{json.dumps(res, indent=2)}\n")
        else:
            print("No images found. Exiting.")
