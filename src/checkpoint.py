# -*- coding: utf-8 -*-
"""모델 체크포인트 저장·복원.

pickle 을 쓰지 않는다(`allow_pickle=False`). 구조는 manifest JSON 문자열 하나와
`param_*`, `BatchNorm*_mean/_var` 배열들이다. 불러올 때는 파일을 믿지 않고
shape·유한성·음수 분산까지 검사한다.
"""

import json
import os
from pathlib import Path

import numpy as np

from network import NeuralNetwork

FORMAT = 1


def save_model(model, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"param_{key}": value for key, value in model.params.items()}
    for name, layer in model.layers.items():
        if name.startswith("BatchNorm"):
            arrays[name + "_mean"] = layer.running_mean
            arrays[name + "_var"] = layer.running_var
    config = {
        key: getattr(model, key)
        for key in [
            "hidden_sizes",
            "use_batchnorm",
            "use_dropout",
            "dropout_ratio",
            "batchnorm_momentum",
        ]
    }
    arrays["manifest"] = np.array(
        json.dumps({"format": FORMAT, "config": config, "metadata": metadata})
    )
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        if manifest["format"] != FORMAT:
            raise ValueError("unsupported model format")
        model = NeuralNetwork(**manifest["config"])
        for key, value in model.params.items():
            saved = data["param_" + key]
            if saved.shape != value.shape or not np.isfinite(saved).all():
                raise ValueError("invalid model weights")
            value[:] = saved
        for name, layer in model.layers.items():
            if name.startswith("BatchNorm"):
                mean, var = data[name + "_mean"], data[name + "_var"]
                if (
                    mean.shape != layer.running_mean.shape
                    or var.shape != layer.running_var.shape
                    or not np.isfinite(mean).all()
                    or not (np.isfinite(var).all() and (var >= 0).all())
                ):
                    raise ValueError("invalid batchnorm statistics")
                layer.running_mean = mean.copy()
                layer.running_var = var.copy()
    return model, manifest["metadata"]
