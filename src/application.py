# -*- coding: utf-8 -*-
"""명령줄 진입점. 실제 동작은 training·checkpoint·images·serving 모듈에 있다.

    python src/application.py demo
    python src/application.py predict examples/digit-7.png
    python src/application.py train | evaluate | serve
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from checkpoint import load_model
from images import recognize
from paths import REFERENCE, ROOT
from serving import serve
from training import evaluate, train

HELP_ON_FAILURE = "Model preparation: make train; use --model .artifacts/model.npz"


def predict(args):
    """이미지 한 장을 판별해 JSON 한 줄로 출력한다."""
    model, _ = load_model(args.model)
    with Image.open(args.image) as image:
        print(json.dumps(recognize(model, image)))


def build_parser():
    parser = argparse.ArgumentParser(description="NumPy single-digit recognizer")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("train", help="새 모델을 학습한다")
    command.add_argument("--output", type=Path, default=ROOT / ".artifacts/model.npz")
    command.add_argument("--epochs", type=int, default=8)
    command.add_argument("--hidden", nargs="+", type=int, default=[128, 64])
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--limit", type=int)
    command.add_argument("--lr", type=float, default=0.001)
    command.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    command.set_defaults(run=train)

    command = commands.add_parser("evaluate", help="공식 test 10,000장으로 평가한다")
    command.add_argument("--model", type=Path, default=REFERENCE)
    command.add_argument("--output")
    command.set_defaults(run=evaluate)

    command = commands.add_parser("predict", help="이미지 파일 한 장을 판별한다")
    command.add_argument("image")
    command.add_argument("--model", type=Path, default=REFERENCE)
    command.set_defaults(run=predict)

    command = commands.add_parser("demo", help="예시 이미지로 한 번 실행한다")
    command.add_argument("--model", type=Path, default=REFERENCE)
    command.set_defaults(run=predict, image=ROOT / "examples/digit-7.png")

    command = commands.add_parser("serve", help="손그림 화면을 띄운다")
    command.add_argument("--model", type=Path, default=REFERENCE)
    command.add_argument("--port", type=int, default=8765)
    command.set_defaults(run=serve)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.run(args)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"{error}\n{HELP_ON_FAILURE}\n")


if __name__ == "__main__":
    main()
