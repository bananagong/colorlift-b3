"""OpenCV HSV baseline, using RGB input and fixed red/yellow/blue mask channels."""
import cv2
import numpy as np
from .config import CAMERAS, COLORS

HSV_RANGES = {
    "red": [((0, 90, 40), (12, 255, 255)), ((168, 90, 40), (179, 255, 255))],
    "yellow": [((18, 90, 40), (38, 255, 255))],
    "blue": [((95, 90, 40), (135, 255, 255))],
}


def hsv_masks(rgb, min_area=5):
    """Return H,W,3 uint8 masks in red/yellow/blue channel order (0 or 255)."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    result = []
    kernel = np.ones((3, 3), np.uint8)
    for color in COLORS:
        mask = np.zeros(rgb.shape[:2], np.uint8)
        for lo, hi in HSV_RANGES[color]:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clean = np.zeros_like(mask)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == label] = 255
        result.append(clean)
    return np.stack(result, axis=-1)


def observation_images(obs):
    return np.stack([obs[camera + "_image"][::-1] for camera in CAMERAS]).copy()


def observation_masks(obs):
    return np.stack([hsv_masks(image) for image in observation_images(obs)])


def mask_preview(masks):
    palette = np.array([[240, 45, 45], [245, 215, 20], [40, 90, 240]], dtype=np.uint8)
    return np.clip((masks.astype(np.float32) / 255) @ palette.astype(np.float32), 0, 255).astype(np.uint8)


def locate_on_table(env, obs, color=None):
    """HSV-only geometric baseline: intersect calibrated image rays with the known cube-center plane.

    No object body position, segmentation ID, depth image, or object ground truth is used.
    """
    from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
    color = color or env.target_color
    masks = observation_masks(obs)
    estimates, details = [], []
    z = env.config.table_height + env.config.cube_size / 2
    for camera, image_masks in zip(CAMERAS, masks):
        mask = image_masks[..., COLORS.index(color)]
        points = cv2.findNonZero(mask)
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        uv = np.array([x + (width - 1) / 2, y + (height - 1) / 2, 1.0])
        K = get_camera_intrinsic_matrix(env.sim, camera, mask.shape[0], mask.shape[1])
        transform = get_camera_extrinsic_matrix(env.sim, camera)
        ray = transform[:3, :3] @ np.linalg.solve(K, uv)
        origin = transform[:3, 3]
        if abs(ray[2]) < 1e-8:
            continue
        point = origin + ((z - origin[2]) / ray[2]) * ray
        estimates.append(point[:2])
        details.append({"camera": camera, "centroid_uv": uv[:2].tolist(), "xy": point[:2].tolist(), "area": int((mask > 0).sum())})
    if not estimates:
        raise RuntimeError("HSV target is not visible in either camera")
    return np.r_[np.mean(estimates, axis=0), z], details
