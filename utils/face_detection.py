"""
AcneVision - Fixed Face Detection v2
Tighter, consistent cropping so Grad-CAM and CNN only see the face.
"""
import cv2
import numpy as np
import os

THIS_FILE = os.path.abspath(__file__)
UTILS_DIR = os.path.dirname(THIS_FILE)
BASE_DIR  = os.path.dirname(UTILS_DIR)

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
PROFILE_PATH = cv2.data.haarcascades + 'haarcascade_profileface.xml'

# Minimum acceptable face area as a fraction of the full image.
# Raised from 0.04 to 0.12 so background-heavy photos are rejected
# or trigger a re-detection at finer settings instead of accepting
# a tiny, loosely-bounded box.
MIN_FACE_AREA_RATIO = 0.12

# Padding around the detected face box. Lower padding keeps the
# crop tight on the face itself rather than including hair,
# shoulders, or background furniture.
CROP_PADDING = 0.08


def detect_face(image_path):
    """
    Detect and tightly crop a face from an image.
    Returns (face_rgb, coords) or (None, None) if no valid face found.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None, None

    h, w = img_bgr.shape[:2]
    print(f"[INFO] Image shape: {img_bgr.shape}")

    face_rgb, coords = _detect_frontal(img_bgr, w, h)
    if face_rgb is not None:
        return face_rgb, coords

    face_rgb, coords = _detect_profile(img_bgr, w, h)
    if face_rgb is not None:
        return face_rgb, coords

    face_rgb, coords = _detect_relaxed(img_bgr, w, h)
    if face_rgb is not None:
        return face_rgb, coords

    face_rgb, coords = _detect_skin_region(img_bgr, w, h)
    if face_rgb is not None:
        return face_rgb, coords

    print("[INFO] No face detected in image")
    return None, None


def _detect_frontal(img_bgr, w, h):
    """Standard frontal face detection — strict settings."""
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray    = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor  = 1.1,
        minNeighbors = 6,
        minSize      = (80, 80),
        maxSize      = (w, h),
        flags        = cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return None, None

    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = face

    face_area  = fw * fh
    image_area = w * h
    if face_area < MIN_FACE_AREA_RATIO * image_area:
        print(f"[INFO] Frontal: detected region too small "
              f"({face_area/image_area:.2%} of image)")
        return None, None

    ratio = fw / fh
    if ratio < 0.6 or ratio > 1.6:
        print(f"[INFO] Frontal: aspect ratio {ratio:.2f} too extreme for a face")
        return None, None

    print(f"[INFO] Frontal face detected at ({x},{y}) size {fw}x{fh}")
    crop = _crop_with_padding(img_bgr, x, y, fw, fh, w, h, padding=CROP_PADDING)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (x, y, fw, fh)


def _detect_profile(img_bgr, w, h):
    """Profile face detection."""
    if not os.path.exists(PROFILE_PATH):
        return None, None

    cascade = cv2.CascadeClassifier(PROFILE_PATH)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray    = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor  = 1.1,
        minNeighbors = 5,
        minSize      = (80, 80),
    )

    if len(faces) == 0:
        return None, None

    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = face

    face_area  = fw * fh
    image_area = w * h
    if face_area < MIN_FACE_AREA_RATIO * image_area:
        return None, None

    print(f"[INFO] Profile face detected at ({x},{y}) size {fw}x{fh}")
    crop = _crop_with_padding(img_bgr, x, y, fw, fh, w, h, padding=CROP_PADDING)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (x, y, fw, fh)


def _detect_relaxed(img_bgr, w, h):
    """
    Relaxed frontal detection for difficult angles.
    Still enforces the same minimum face-area ratio as the strict
    pass, so a 'relaxed' detection cannot slip through with a
    tiny box surrounded by background.
    """
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray    = cv2.equalizeHist(gray)

    best_face = None
    best_area = 0

    for scale in [1.05, 1.15, 1.2]:
        for neighbors in [4, 3]:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor  = scale,
                minNeighbors = neighbors,
                minSize      = (60, 60),
            )
            if len(faces) == 0:
                continue

            for f in faces:
                x, y, fw, fh = f
                area  = fw * fh
                ratio = fw / fh

                if area < MIN_FACE_AREA_RATIO * w * h:
                    continue
                if ratio < 0.55 or ratio > 1.8:
                    continue
                if area > best_area:
                    best_area = area
                    best_face = (x, y, fw, fh, scale, neighbors)

    if best_face is None:
        return None, None

    x, y, fw, fh, scale, neighbors = best_face
    print(f"[INFO] Relaxed face detected (scale={scale}, neighbors={neighbors}) "
          f"size {fw}x{fh}")
    crop = _crop_with_padding(img_bgr, x, y, fw, fh, w, h, padding=CROP_PADDING)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (x, y, fw, fh)


def _detect_skin_region(img_bgr, w, h):
    """
    Fallback: detect large skin-coloured region.
    Tightened to require a larger, more face-shaped region
    before accepting, to avoid wide shots of hands/necks/background.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([0,  20, 70],  dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(img_hsv, lower, upper)

    lower2 = np.array([0, 10, 50],  dtype=np.uint8)
    upper2 = np.array([35, 200, 255], dtype=np.uint8)
    mask2  = cv2.inRange(img_hsv, lower2, upper2)
    mask   = cv2.bitwise_or(mask, mask2)

    kernel = np.ones((15, 15), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    skin_ratio = np.sum(mask > 0) / (w * h)

    if skin_ratio < 0.20:
        print(f"[INFO] Skin region: {skin_ratio:.1%} of image — too small, rejecting")
        return None, None

    coords = cv2.findNonZero(mask)
    if coords is None:
        return None, None

    rx, ry, rw, rh = cv2.boundingRect(coords)

    # Require the skin box to be reasonably square (face-shaped),
    # not a wide strip across the whole frame
    box_ratio = rw / rh if rh > 0 else 0
    if rw < w * 0.25 or rh < h * 0.25:
        print(f"[INFO] Skin bounding box too small: {rw}x{rh}")
        return None, None
    if box_ratio > 2.2 or box_ratio < 0.35:
        print(f"[INFO] Skin bounding box not face-shaped (ratio={box_ratio:.2f})")
        return None, None

    print(f"[INFO] Skin region fallback: {skin_ratio:.1%} of image")
    crop = _crop_with_padding(img_bgr, rx, ry, rw, rh, w, h, padding=0.05)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (rx, ry, rw, rh)


def _crop_with_padding(img, x, y, fw, fh, W, H, padding=0.08):
    """
    Crop face with light padding, clamped to image bounds.
    Padding default lowered from 0.15 to 0.08 so the resulting
    crop stays tightly focused on the face rather than pulling
    in surrounding hair, neck, or background.
    """
    pad_x = int(fw * padding)
    pad_y = int(fh * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + fw + pad_x)
    y2 = min(H, y + fh + pad_y)
    return img[y1:y2, x1:x2]