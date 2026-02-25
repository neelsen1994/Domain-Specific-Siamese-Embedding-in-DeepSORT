"""OpenCV-based data augmentation pipeline for turkey ReID.

All transforms operate on uint8 BGR images, consistent with the DeepSORT
pipeline that also works in BGR color space.

Augmentations included (all tuned for barn / outdoor livestock imagery):
  1. Random horizontal flip
  2. Random crop-with-reflect-padding  (replaces random-resize-crop)
  3. Random brightness + contrast jitter
  4. Random Gaussian blur
  5. Random rotation
  6. Random zoom
  7. Random erasing / cutout  (occlusion simulation)
"""
from __future__ import annotations

import random
from typing import Optional, Tuple

import cv2
import numpy as np


class ReIDAugmentation:
    """Randomised augmentation pipeline for a single BGR uint8 image.

    Parameters
    ----------
    image_size  : (H, W) – output size (same as input after crop)
    p_flip      : probability of horizontal flip
    p_rotation  : probability of random rotation
    max_rotation: maximum rotation angle in degrees
    p_zoom      : probability of random zoom
    zoom_range  : zoom range as (min_scale, max_scale)
    p_erasing   : probability of random erasing
    p_blur      : probability of Gaussian blur
    brightness  : maximum absolute brightness delta (0–255 scale)
    contrast    : maximum relative contrast multiplier deviation
    seed        : optional seed for reproducibility; call before training
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 64),
        p_flip: float = 0.5,
        p_rotation: float = 0.05,
        max_rotation: float = 15.0,
        p_zoom: float = 0.1,
        zoom_range: Tuple[float, float] = (0.9, 1.1),
        p_erasing: float = 0.40,
        p_blur: float = 0.20,
        brightness: float = 30.0,
        contrast: float = 0.30,
        seed: Optional[int] = None,
    ):
        self.H, self.W = image_size
        self.p_flip = p_flip
        self.p_rotation = p_rotation
        self.max_rotation = max_rotation
        self.p_zoom = p_zoom
        self.zoom_range = zoom_range
        self.p_erasing = p_erasing
        self.p_blur = p_blur
        self.brightness = brightness
        self.contrast = contrast
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply the augmentation pipeline to a single image.

        Parameters
        ----------
        img : (H, W, 3) uint8 BGR

        Returns
        -------
        aug : (H, W, 3) uint8 BGR  (same spatial size)
        """
        img = img.copy()

        # 1. Horizontal flip
        if random.random() < self.p_flip:
            img = cv2.flip(img, 1)

        # 2. Random rotation
        if random.random() < self.p_rotation:
            img = self._random_rotation(img)

        # 3. Random zoom
        if random.random() < self.p_zoom:
            img = self._random_zoom(img)

        # 4. Crop-with-reflect-padding (8 px each side)
        img = self._random_crop_pad(img, pad=8)

        # 5. Brightness + contrast jitter
        img = self._color_jitter(img)

        # 6. Gaussian blur
        if random.random() < self.p_blur:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 7. Random erasing
        if random.random() < self.p_erasing:
            img = self._random_erase(img)

        return img

    # ------------------------------------------------------------------

    def _random_crop_pad(self, img: np.ndarray, pad: int) -> np.ndarray:
        """Pad by *pad* pixels (reflect-101) then randomly crop back to H×W."""
        padded = cv2.copyMakeBorder(
            img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101
        )
        ph, pw = padded.shape[:2]
        max_y  = ph - self.H
        max_x  = pw - self.W
        y0 = random.randint(0, max_y)
        x0 = random.randint(0, max_x)
        return padded[y0: y0 + self.H, x0: x0 + self.W]

    def _random_rotation(self, img: np.ndarray) -> np.ndarray:
        """Apply random rotation with specified angle range.

        Rotated image is cropped/padded back to original size.
        """
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rotation_matrix, (w, h),
                                 borderMode=cv2.BORDER_REFLECT_101)
        return rotated

    def _random_zoom(self, img: np.ndarray) -> np.ndarray:
        """Apply random zoom by scaling the image.

        Zoomed image is cropped/padded back to original size.
        """
        zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
        h, w = img.shape[:2]

        # Resize to zoomed size
        new_h = int(h * zoom_factor)
        new_w = int(w * zoom_factor)
        zoomed = cv2.resize(img, (new_w, new_h))

        # Crop or pad to get back to original size
        if zoom_factor > 1.0:
            # Crop from center
            y0 = (new_h - h) // 2
            x0 = (new_w - w) // 2
            result = zoomed[y0: y0 + h, x0: x0 + w]
        else:
            # Pad with reflect border
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            result = cv2.copyMakeBorder(
                zoomed, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101
            )
            # Handle odd padding
            if result.shape[0] != h or result.shape[1] != w:
                result = result[:h, :w]

        return result

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """Per-image brightness and contrast adjustment."""
        img = img.astype(np.float32)
        b = random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        img = img * c + b
        return np.clip(img, 0, 255).astype(np.uint8)

    def _random_erase(
        self,
        img: np.ndarray,
        sl: float = 0.02,
        sh: float = 0.30,
        min_ratio: float = 0.30,
        max_ratio: float = 3.30,
        fill: str = "mean",
        attempts: int = 100,
    ) -> np.ndarray:
        """Random erasing (Zhong et al. 2017, AAAI 2020).

        Replaces a random rectangular region with the image mean (or random
        noise) to simulate partial occlusion of the animal.
        """
        H, W = img.shape[:2]
        area = H * W
        for _ in range(attempts):
            erase_area = random.uniform(sl, sh) * area
            aspect     = random.uniform(min_ratio, max_ratio)
            eh = int(round((erase_area * aspect) ** 0.5))
            ew = int(round((erase_area / aspect) ** 0.5))
            if eh >= H or ew >= W:
                continue
            y0 = random.randint(0, H - eh)
            x0 = random.randint(0, W - ew)
            if fill == "mean":
                fill_val = img.mean(axis=(0, 1)).astype(np.uint8)
            else:
                fill_val = np.random.randint(0, 256, (3,), dtype=np.uint8)
            img[y0: y0 + eh, x0: x0 + ew] = fill_val
            break
        return img
