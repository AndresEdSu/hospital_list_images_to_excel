from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps


def _order_corners(points: object) -> object:
    import numpy as np

    values = np.asarray(points, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")
    sums = values.sum(axis=1)
    differences = np.diff(values, axis=1).reshape(-1)
    ordered[0] = values[sums.argmin()]
    ordered[2] = values[sums.argmax()]
    ordered[1] = values[differences.argmin()]
    ordered[3] = values[differences.argmax()]
    return ordered


def _correct_perspective(image: Image.Image) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return image

    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    image_area = width * height
    corners = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        if cv2.contourArea(contour) < image_area * 0.55:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 4:
            candidate = _order_corners(approximation)
            margin_x = width * 0.15
            margin_y = height * 0.15
            near_edges = (
                candidate[0][0] <= margin_x
                and candidate[0][1] <= margin_y
                and candidate[1][0] >= width - margin_x
                and candidate[1][1] <= margin_y
                and candidate[2][0] >= width - margin_x
                and candidate[2][1] >= height - margin_y
                and candidate[3][0] <= margin_x
                and candidate[3][1] >= height - margin_y
            )
            if near_edges:
                corners = candidate
                break
    if corners is None:
        return image

    top_left, top_right, bottom_right, bottom_left = corners
    target_width = int(
        max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
    )
    target_height = int(
        max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
    )
    if target_width < 300 or target_height < 300:
        return image
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    corrected = cv2.warpPerspective(
        rgb,
        transform,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(corrected)


def _normalize_shadows(image: Image.Image) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return image

    rgb = np.asarray(image)
    sigma = max(rgb.shape[:2]) / 35
    channels = cv2.split(rgb)
    normalized_channels = []
    for channel in channels:
        background = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma)
        background = np.maximum(background, 1)
        normalized_channels.append(cv2.divide(channel, background, scale=245))
    normalized = cv2.merge(normalized_channels)
    blended = cv2.addWeighted(rgb, 0.25, normalized, 0.75, 0)
    return Image.fromarray(blended)


def preprocess_image(
    source: Path,
    destination: Path,
    minimum_long_side: int = 1600,
) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = _correct_perspective(image)
        image = _normalize_shadows(image)
        long_side = max(image.size)
        if long_side < minimum_long_side:
            scale = minimum_long_side / long_side
            target = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            image = image.resize(target, Image.Resampling.LANCZOS)
        image = ImageOps.autocontrast(image, cutoff=1)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
        image.save(destination, format="JPEG", quality=95, optimize=True)
        return image.size
