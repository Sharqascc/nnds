"""
Check source image(s) for EXIF focal-length metadata, and compute what
that implies in pixels, for comparison against the guessed f = image_width
assumption used in pnp_vs_homography_loocv.py.
"""
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "Pillow"], check=True)
    from PIL import Image
    from PIL.ExifTags import TAGS


def find_candidate_images(root="/content/nnds_verify"):
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic")
    root = Path(root)
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def read_exif(path):
    img = Image.open(path)
    exif_raw = img._getexif() if hasattr(img, "_getexif") else None
    if not exif_raw:
        return None, img.size
    tags = {}
    for tag_id, value in exif_raw.items():
        tag_name = TAGS.get(tag_id, tag_id)
        tags[tag_name] = value
    return tags, img.size


def main():
    images = find_candidate_images()
    print(f"Found {len(images)} candidate image file(s) under /content/nnds_verify")
    for p in images:
        print(f"\n{'='*80}\n{p}\n{'='*80}")
        try:
            tags, (w, h) = read_exif(p)
        except Exception as e:
            print(f"  could not read: {e}")
            continue

        print(f"  image size: {w}x{h}")
        if not tags:
            print("  no EXIF data found (image likely stripped of metadata, "
                  "screenshot, or re-encoded)")
            continue

        focal_mm = tags.get("FocalLength")
        focal_35mm = tags.get("FocalLengthIn35mmFilm")
        model = tags.get("Model")
        make = tags.get("Make")

        print(f"  Make/Model:            {make} {model}")
        print(f"  FocalLength (mm):      {focal_mm}")
        print(f"  FocalLengthIn35mmFilm: {focal_35mm}")

        if focal_35mm:
            f_px = float(focal_35mm) * w / 36.0
            print(f"  --> estimated f_pixels (via 35mm-equiv): {f_px:.1f}")
            print(f"      (current script assumes f_pixels = {w} = image width)")
        else:
            print("  --> no 35mm-equivalent focal length available; "
                  "would need sensor width + FocalLength(mm) instead, "
                  "which EXIF often omits for phone cameras")


if __name__ == "__main__":
    main()
