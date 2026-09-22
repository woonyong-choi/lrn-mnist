# -*- coding: utf-8 -*-
"""입력 이미지를 MNIST 규약의 28x28 벡터로 바꾸고 한 장을 판별한다."""

import numpy as np
from PIL import Image


def preprocess_image(image):
    image = image.convert("L")
    if image.width > 4096 or image.height > 4096:
        raise ValueError("image exceeds 4096 pixels per side")
    array = np.asarray(image, dtype=np.uint8)
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
    if np.median(border) > 127:
        array = 255 - array
    ys, xs = np.nonzero(array > 25)
    if not len(xs):
        raise ValueError("blank input: draw one digit")
    crop = Image.fromarray(array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
    scale = 20 / max(crop.size)
    crop = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (28, 28))
    canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32)
    y, x = np.indices(pixels.shape)
    mass = pixels.sum()
    dy = round(13.5 - (y * pixels).sum() / mass)
    dx = round(13.5 - (x * pixels).sum() / mass)
    centered = Image.new("L", (28, 28))
    centered.paste(canvas, (dx, dy))
    return np.asarray(centered, dtype=np.float32).reshape(1, 784) / 255.0

def recognize(model, image):
    pixels = preprocess_image(image)
    scores = model.predict(pixels)[0]
    return {
        "digit": int(scores.argmax()),
        "scores": scores.tolist(),
        "pixels": np.rint(pixels.reshape(28, 28) * 255).astype(int).tolist(),
        "note": "Class probabilities are not calibrated confidence; drawings differ from MNIST.",
    }
