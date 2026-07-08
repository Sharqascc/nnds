"""
Probe the source traffic video file for container-level metadata
(codec, encoder, device tags, creation info) using ffprobe, since the
individual extracted JPEG frames had their EXIF stripped. Also lists
sibling files in sample_data/ in case a camera-info sidecar file exists.
"""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/content/nnds_verify")
VIDEO_PATH = ROOT / "sample_data" / "traffic_video.mp4"


def ensure_ffprobe():
    if shutil.which("ffprobe") is None:
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)


def probe(path):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffprobe failed:", result.stderr)
        return None
    return json.loads(result.stdout)


def main():
    print("=" * 80)
    print(f"Checking sample_data/ directory contents")
    print("=" * 80)
    sd = ROOT / "sample_data"
    if sd.exists():
        for p in sorted(sd.iterdir()):
            print(f"  {p.name}  ({p.stat().st_size} bytes)")
    else:
        print("  sample_data/ does not exist")

    print("\n" + "=" * 80)
    print(f"Probing {VIDEO_PATH}")
    print("=" * 80)
    if not VIDEO_PATH.exists():
        print("  video file not found at expected path")
        return

    ensure_ffprobe()
    info = probe(VIDEO_PATH)
    if info is None:
        return

    fmt = info.get("format", {})
    print("\n--- format ---")
    print(f"  filename:    {fmt.get('filename')}")
    print(f"  format_name: {fmt.get('format_name')}")
    print(f"  duration:    {fmt.get('duration')} s")
    print(f"  size:        {fmt.get('size')} bytes")
    tags = fmt.get("tags", {})
    if tags:
        print("  tags:")
        for k, v in tags.items():
            print(f"    {k}: {v}")
    else:
        print("  no format-level tags")

    for i, stream in enumerate(info.get("streams", [])):
        print(f"\n--- stream {i} ({stream.get('codec_type')}) ---")
        for key in ("codec_name", "width", "height", "avg_frame_rate", "r_frame_rate"):
            if key in stream:
                print(f"  {key}: {stream[key]}")
        stags = stream.get("tags", {})
        if stags:
            print("  tags:")
            for k, v in stags.items():
                print(f"    {k}: {v}")
        else:
            print("  no stream-level tags")

    print("\nDone.")


if __name__ == "__main__":
    main()
