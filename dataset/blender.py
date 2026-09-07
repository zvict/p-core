"""Loader for NeRF/Blender ``transforms_{split}.json`` datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def load_blender_data(
    root: str | Path,
    split: str = "train",
    factor: int = 1,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, float], list[str]]:
    """Load images and camera poses while preserving the original preprocessing."""
    root = Path(root)
    with (root / f"transforms_{split}.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    images: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    paths: list[str] = []
    for frame in metadata["frames"]:
        relative = Path(frame["file_path"])
        if not relative.suffix:
            relative = relative.with_suffix(".png")
        path = Path(os.path.abspath(root / relative))
        image = imageio.imread(path)
        height, width = image.shape[:2]
        if factor > 1:
            # No explicit filter is intentional: this is the legacy preprocessing.
            image = np.asarray(Image.fromarray(image).resize((width // factor, height // factor)))
        images.append((image / 255.0).astype(np.float32))
        poses.append(np.asarray(frame["transform_matrix"], dtype=np.float32))
        paths.append(str(path))

    image_array = np.asarray(images, dtype=np.float32)
    pose_array = np.asarray(poses, dtype=np.float32)
    height, width = image_array[0].shape[:2]
    focal = 0.5 * width / np.tan(0.5 * float(metadata["camera_angle_x"]))
    return image_array, pose_array, (height, width, focal), paths
