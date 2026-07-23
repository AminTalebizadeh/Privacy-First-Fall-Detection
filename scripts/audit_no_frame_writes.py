#!/usr/bin/env python3
"""Static privacy audit: fail if any source file contains an
image/video-writing call. Run in CI:  python scripts/audit_no_frame_writes.py

This complements the runtime schema test — the acceptance criterion is
"no code path writes raw frames to disk", so we enforce it at the
source level too.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN = [
    r"cv2\.imwrite", r"cv2\.VideoWriter", r"imageio\.(imwrite|mimsave)",
    r"\.save\(\s*['\"].*\.(png|jpg|jpeg|bmp|tiff)", r"plt\.imsave",
    r"PIL\.Image", r"skimage\.io\.imsave", r"torchvision\.utils\.save_image",
]

ROOT = Path(__file__).resolve().parents[1]
SCAN = [ROOT / "src", ROOT / "scripts"]
SELF = Path(__file__).resolve()


def main() -> int:
    pattern = re.compile("|".join(FORBIDDEN))
    hits = []
    for base in SCAN:
        for py in base.rglob("*.py"):
            if py.resolve() == SELF:
                continue
            for i, line in enumerate(py.read_text().splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{py.relative_to(ROOT)}:{i}: {line.strip()}")
    if hits:
        print("PRIVACY AUDIT FAILED — image-writing code found:")
        print("\n".join(hits))
        return 1
    print("privacy audit OK: no image-writing code paths in src/ or scripts/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
