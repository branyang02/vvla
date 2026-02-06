"""HuggingFace Hub utilities for policy serialization."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from vvla.transforms.transforms import NormStats


def norm_stats_to_dict(norm_stats: dict[str, NormStats]) -> dict[str, dict[str, Any]]:
    """Convert NormStats dict to JSON-serializable format."""
    result = {}
    for key, stats in norm_stats.items():
        result[key] = {
            "mean": stats.mean.tolist(),
            "std": stats.std.tolist(),
            "q01": stats.q01.tolist() if stats.q01 is not None else None,
            "q99": stats.q99.tolist() if stats.q99 is not None else None,
        }
    return result


def dict_to_norm_stats(data: dict[str, dict[str, Any]]) -> dict[str, NormStats]:
    """Reconstruct NormStats dict from JSON data."""
    result = {}
    for key, stats_dict in data.items():
        result[key] = NormStats(
            mean=np.array(stats_dict["mean"], dtype=np.float32),
            std=np.array(stats_dict["std"], dtype=np.float32),
            q01=np.array(stats_dict["q01"], dtype=np.float32)
            if stats_dict["q01"]
            else None,
            q99=np.array(stats_dict["q99"], dtype=np.float32)
            if stats_dict["q99"]
            else None,
        )
    return result


def serialize_transforms(transforms: list[Any]) -> list[dict[str, Any]]:
    """Serialize a list of transform objects to JSON-serializable dicts."""
    result = []
    for t in transforms:
        t_dict: dict[str, Any] = {"_type": type(t).__name__}
        if hasattr(t, "norm_stats"):
            t_dict["norm_stats"] = norm_stats_to_dict(t.norm_stats)
        if hasattr(t, "height"):
            t_dict["height"] = t.height
        if hasattr(t, "width"):
            t_dict["width"] = t.width
        result.append(t_dict)
    return result


def deserialize_transforms(data: list[dict[str, Any]]) -> list[Any]:
    """Reconstruct transform objects from JSON data."""
    from vvla.transforms.transforms import Normalize, ResizeTransform, Unnormalize

    transform_classes: dict[str, type] = {
        "Normalize": Normalize,
        "Unnormalize": Unnormalize,
        "ResizeTransform": ResizeTransform,
    }

    result = []
    for t_dict in data:
        cls_name = t_dict["_type"]
        if cls_name not in transform_classes:
            raise ValueError(f"Unknown transform class: {cls_name}")

        cls = transform_classes[cls_name]

        if cls_name in ("Normalize", "Unnormalize"):
            norm_stats = dict_to_norm_stats(t_dict["norm_stats"])
            result.append(cls(norm_stats=norm_stats))
        elif cls_name == "ResizeTransform":
            result.append(
                cls(height=t_dict.get("height", 224), width=t_dict.get("width", 224))
            )
        else:
            result.append(cls())

    return result


def save_transforms_json(
    input_transforms: list[Any],
    output_transforms: list[Any],
    path: Path,
) -> None:
    """Save transforms to transforms.json."""
    data = {
        "input_transforms": serialize_transforms(input_transforms),
        "output_transforms": serialize_transforms(output_transforms),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_transforms_json(path: Path) -> tuple[list[Any], list[Any]]:
    """Load transforms from transforms.json.

    Returns:
        Tuple of (input_transforms, output_transforms).
    """
    with open(path) as f:
        data = json.load(f)
    return (
        deserialize_transforms(data["input_transforms"]),
        deserialize_transforms(data["output_transforms"]),
    )
