"""便捷入口：python main.py --material ./material --bgm ./bgm --output ./output/final.mp4"""
from __future__ import annotations

import sys

from autocut.cli import main

if __name__ == "__main__":
    sys.exit(main())
