"""前処理packageをimportせず、既存評価と同じ主ラベルを選ぶ。"""
from __future__ import annotations

import numpy as np


def primary_label(labels: dict[str, np.ndarray], dataset_name: str) -> np.ndarray | None:
    if dataset_name in ("coil20", "coil100"):
        return labels.get("object_id")
    if dataset_name == "pancreas":
        return labels.get("celltype")
    if dataset_name == "mock":
        return labels.get("cluster")
    if dataset_name == "fashion_mnist":
        return labels.get("class")
    if dataset_name == "twenty_newsgroups":
        return labels.get("newsgroup")
    if dataset_name == "mnist_oos":
        return labels.get("digit")
    if dataset_name == "hong_ed":
        return labels.get("admitted")
    if dataset_name in ("organoid", "embryoid_body"):
        return next(iter(labels.values())) if labels else None
    return None
