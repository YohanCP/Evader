import cv2
import numpy as np

def overlay_transparent(bg, overlay, x, y, overlay_size=None):
    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]

    # Ensure overlay fits within the background dimensions
    if x + w > bg.shape[1] or y + h > bg.shape[0] or x < 0 or y < 0:
        return bg

    # Extract overlay components
    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3] / 255.0
    mask = np.stack([mask] * 3, axis=-1)

    # Apply blending
    roi = bg[y:y+h, x:x+w]
    blended = (1.0 - mask) * roi + mask * overlay_img
    bg[y:y+h, x:x+w] = blended.astype(np.uint8)

    return bg