"""원격 refine API의 4096px/16MP 제한에 맞춘 원본 캐시를 만든다."""

from pathlib import Path
from PIL import Image, ImageOps

root = Path(__file__).resolve().parents[1]
source_dir = root / "assets" / "pinterest-closet-sources"
output_dir = root / "assets" / "pinterest-refine-sources"
output_dir.mkdir(exist_ok=True)

for index, source in enumerate(sorted(source_dir.iterdir()), 1):
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((3840, 3840), Image.Resampling.LANCZOS)
        image.save(output_dir / f"{source.stem}.jpg", "JPEG", quality=92, optimize=True)
    print(f"\rrefine 원본 준비 {index}/100", end="", flush=True)
print()
