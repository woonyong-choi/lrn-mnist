# -*- coding: utf-8 -*-
"""저장소 안의 고정 경로. 실행 위치와 무관하게 같은 파일을 가리킨다."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "models/reference.npz"
DATA_DIR = ROOT / "data"
WEB_PAGE = ROOT / "web/index.html"
