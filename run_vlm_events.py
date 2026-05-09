from pathlib import Path
import argparse

from vlm_events import VLMEventsAnnotator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pet-csv", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    annotator = VLMEventsAnnotator()
    annotator.annotate_pets(
        pet_csv=Path(args.pet_csv),
        video_path=Path(args.video),
        out_csv=Path(args.out_csv),
    )

if __name__ == "__main__":
    main()
