"""Write Tests/SAM31Tests/Resources/resize-golden.json for PreprocessTests.

Each case is an RGB noise image from the 64-bit LCG below (mirrored by `PreprocessTests.noise`),
resized with Pillow's BILINEAR (the resize in
mlx_vlm/models/sam3/processing_sam3.py::_process_single_image). Large outputs are stored as the
SHA-256 of their (H, W, 3) uint8 bytes; small ones as base64 bytes, so a failure shows the diff.

    .venv/bin/python parity/make_resize_golden.py
"""

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

MASK = (1 << 64) - 1

# name: (width, height, out width, out height, seed)
CASES = {
    "small": (37, 23, 1008, 1008, 1),
    "hd": (1920, 1080, 1008, 1008, 2),
    "sd": (640, 360, 1008, 1008, 3),
    "column": (1, 57, 7, 13, 4),  # 1xN
    "row": (61, 1, 11, 3, 5),  # Nx1
    "tall": (3, 400, 5, 7, 6),  # >100x taller than wide: Pillow resizes vertically first
    "identity": (13, 9, 13, 9, 7),
    "upscale": (5, 4, 17, 11, 8),
    "downscale": (97, 41, 23, 17, 9),  # non-square
    "width_only": (50, 20, 20, 20, 10),  # vertical pass skipped
}
RAW_MAX_BYTES = 4096


def noise(count: int, seed: int) -> np.ndarray:
    """state = state * 6364136223846793005 + 1442695040888963407 (mod 2^64); byte = state >> 56."""
    state = (0x9E3779B97F4A7C15 ^ seed) & MASK
    out = bytearray(count)
    for i in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK
        out[i] = state >> 56
    return np.frombuffer(bytes(out), dtype=np.uint8)


cases = []
for name, (w, h, ow, oh, seed) in CASES.items():
    image = noise(w * h * 3, seed).reshape(h, w, 3)
    out = np.ascontiguousarray(np.asarray(Image.fromarray(image).resize((ow, oh), Image.BILINEAR)))
    case = {"name": name, "width": w, "height": h, "outWidth": ow, "outHeight": oh, "seed": seed}
    if out.nbytes <= RAW_MAX_BYTES:
        case["output"] = base64.b64encode(out.tobytes()).decode()
    else:
        case["sha256"] = hashlib.sha256(out.tobytes()).hexdigest()
    cases.append(case)

path = Path(__file__).resolve().parent.parent / "Tests/SAM31Tests/Resources/resize-golden.json"
path.write_text(json.dumps({"pillow": PIL.__version__, "cases": cases}, indent=1) + "\n")
print(f"wrote {path} ({path.stat().st_size} bytes, Pillow {PIL.__version__})")
