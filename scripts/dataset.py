"""Dataset loading, splitting, and PK batch sampling for turkey ReID training.

Folder-per-identity structure is assumed:
    dataset_root/
        object_1/   frame_100.jpg  frame_200.jpg  ...
        object_2/   ...
        ...
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_dict(dataset_path: str) -> Dict[str, List[str]]:
    """Load a folder-per-identity dataset.

    Returns
    -------
    dict mapping class_name (str) -> sorted list of image paths
    """
    dataset: Dict[str, List[str]] = {}
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError("Dataset path not found: {}".format(dataset_path))

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        images = sorted(
            str(p) for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if images:
            dataset[class_dir.name] = images

    if not dataset:
        raise ValueError("No identities found in {}".format(dataset_path))

    print("[dataset] Loaded {} identities from {}".format(len(dataset), dataset_path))
    imgs_per_id = [len(v) for v in dataset.values()]
    print("[dataset] Images/ID: min={} max={} mean={:.1f}".format(
        min(imgs_per_id), max(imgs_per_id), float(sum(imgs_per_id)) / len(imgs_per_id)))
    return dataset


# ---------------------------------------------------------------------------
# Deterministic train / val / test split (by image, within each identity)
# ---------------------------------------------------------------------------

def create_split(
    dataset: Dict[str, List[str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    manifest_path: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Split images within each identity into train / val / test.

    Every identity appears in all three splits so the model can be evaluated
    on same-identity retrieval (closed-set ReID).  A fixed *seed* guarantees
    exact reproducibility.  Borrowing from train prevents empty splits for
    low-image identities.

    Parameters
    ----------
    dataset       : {class_id: [image_paths]}
    train_ratio   : fraction for training (default 0.70)
    val_ratio     : fraction for validation (default 0.15)
    seed          : random seed
    manifest_path : if not None, save the split to this JSON path

    Returns
    -------
    (train_data, val_data, test_data)  – same dict type as *dataset*
    """
    rng = random.Random(seed)
    train_data: Dict[str, List[str]] = {}
    val_data: Dict[str, List[str]] = {}
    test_data: Dict[str, List[str]] = {}

    for class_id, images in dataset.items():
        shuffled = images[:]
        rng.shuffle(shuffled)
        n = len(shuffled)

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(0, int(round(n * val_ratio)))

        # Ensure at least 1 image in each split when possible
        if n >= 3 and n_train + n_val >= n:
            n_train = n - 2
            n_val = 1
        elif n == 2 and n_train + n_val >= n:
            n_train = 1
            n_val = 0
        # n == 1: n_train=1, n_val=0, test borrows from train

        train_data[class_id] = shuffled[:n_train]
        val_data[class_id] = shuffled[n_train: n_train + n_val]
        test_data[class_id] = shuffled[n_train + n_val:]

        # Safety fallback: borrow from train when a split is empty
        if not val_data[class_id]:
            val_data[class_id] = train_data[class_id][:1]
        if not test_data[class_id]:
            test_data[class_id] = train_data[class_id][:1]

    if manifest_path:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        manifest = {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "splits": {
                "train": train_data,
                "val": val_data,
                "test": test_data,
            },
        }
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print("[dataset] Split manifest saved -> {}".format(manifest_path))

    _print_split_stats(train_data, val_data, test_data)
    return train_data, val_data, test_data


def load_split_from_manifest(
    manifest_path: str,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Reload a previously saved split manifest (JSON)."""
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    s = manifest["splits"]
    print("[dataset] Loaded split from {}".format(manifest_path))
    _print_split_stats(s["train"], s["val"], s["test"])
    return s["train"], s["val"], s["test"]


def _print_split_stats(
    train: Dict[str, List[str]],
    val: Dict[str, List[str]],
    test: Dict[str, List[str]],
) -> None:
    print(
        "[dataset] {} IDs | train={} val={} test={} images".format(
            len(train),
            sum(len(v) for v in train.values()),
            sum(len(v) for v in val.values()),
            sum(len(v) for v in test.values()),
        )
    )


# ---------------------------------------------------------------------------
# PK Sampler  (P identities x K images per batch)
# ---------------------------------------------------------------------------

class PKSampler:
    """Yield batches of P identities × K images for batch-hard triplet training.

    With-replacement sampling lets us always draw K images per identity even
    when an identity has fewer than K images (common in small turkey datasets).

    Parameters
    ----------
    data       : {class_id: [image_paths]}
    P          : number of identities per batch
    K          : images per identity per batch  →  batch_size = P * K
    image_size : (H, W) in pixels
    augment_fn : optional callable (uint8 BGR array) -> uint8 BGR array
    seed       : numpy random seed
    """

    def __init__(
        self,
        data: Dict[str, List[str]],
        P: int = 16,
        K: int = 4,
        image_size: Tuple[int, int] = (128, 64),
        augment_fn=None,
        seed: int = 42,
    ):
        self.data = {k: v for k, v in data.items() if v}
        if len(self.data) < 2:
            raise ValueError(
                "Need at least 2 identities for triplet training "
                "(got {}).".format(len(self.data))
            )
        self.class_ids = list(self.data.keys())
        self.P = P
        self.K = K
        self.H, self.W = image_size
        self.augment_fn = augment_fn
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        """Approximate steps per epoch based on total training images."""
        total = sum(len(v) for v in self.data.values())
        return max(1, total // (self.P * self.K))

    # ------------------------------------------------------------------

    def _load(self, path: str) -> np.ndarray:
        """Load, resize; return uint8 BGR [0..255]."""
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.W, self.H))  # cv2 wants (W, H)
        return img

    # ------------------------------------------------------------------

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return one PK batch.

        Returns
        -------
        images : (P*K, H, W, 3) float32  raw BGR [0, 255]
        labels : (P*K,)         int32    consecutive labels 0..P-1
        """
        n_cls = len(self.class_ids)
        P = min(self.P, n_cls)
        chosen = self.rng.choice(n_cls, size=P, replace=False)

        images: List[np.ndarray] = []
        labels: List[int] = []

        for lbl, cidx in enumerate(chosen):
            paths = self.data[self.class_ids[cidx]]
            # Sample K images; use replacement when fewer than K images exist
            idxs = self.rng.choice(len(paths), size=self.K,
                                   replace=len(paths) < self.K)
            for i in idxs:
                img = self._load(paths[i])
                if self.augment_fn is not None:
                    img = self.augment_fn(img)
                images.append(img.astype(np.float32))
                labels.append(lbl)

        return np.stack(images), np.array(labels, dtype=np.int32)

    # ------------------------------------------------------------------

    def epoch_generator(self, steps: Optional[int] = None):
        """Generator that yields *steps* batches (one epoch).

        Yields
        ------
        (images, labels) tuples – see sample_batch()
        """
        n = steps if steps is not None else len(self)
        for _ in range(n):
            yield self.sample_batch()


# ---------------------------------------------------------------------------
# Evaluation array builder
# ---------------------------------------------------------------------------

def load_split_as_arrays(
    data: Dict[str, List[str]],
    image_size: Tuple[int, int] = (128, 64),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all images from a split dict into numpy arrays.

    Returns
    -------
    images     : (N, H, W, 3) float32  raw BGR [0, 255]
    labels     : (N,)         int32    integer class indices
    class_names: list of class names ordered by label index
    """
    class_names = sorted(data.keys())
    c2i = {c: i for i, c in enumerate(class_names)}
    H, W = image_size

    imgs: List[np.ndarray] = []
    lbls: List[int] = []

    for cls, paths in data.items():
        idx = c2i[cls]
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.resize(img, (W, H)).astype(np.float32)
            imgs.append(img)
            lbls.append(idx)

    if not imgs:
        raise ValueError("No images could be loaded from the split.")

    return np.stack(imgs), np.array(lbls, dtype=np.int32), class_names
