"""
Lesion Detector - Fixed Version
=================================
- Ignores hair regions
- Only analyses actual skin-colored pixels
- More accurate feature detection
"""

import cv2
import numpy as np


COLORS = {
    'Pustule'          : (0,   200, 80),
    'Papule'           : (60,  60,  220),
    'Blackhead'        : (200, 80,  0),
    'Whitehead'        : (0,   220, 220),
    'Redness'          : (80,  80,  255),
    'Hyperpigmentation': (0,   140, 255),
    'Rough Texture'    : (200, 0,   200),
}


def get_skin_mask(img_rgb):
    """
    Create a mask that isolates ONLY skin pixels.
    Excludes hair, background, clothing, lips, eyes.
    Uses YCrCb color space which is best for skin detection.
    """
    # Convert to YCrCb — best color space for skin
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

    # Skin color range in YCrCb
    # Y=luminance, Cr=red chroma, Cb=blue chroma
    lower_skin = np.array([0,  133, 77],  dtype=np.uint8)
    upper_skin = np.array([255,173, 127], dtype=np.uint8)
    skin_mask  = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Also use HSV for better skin isolation
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_hsv = np.array([0,  15, 50],  dtype=np.uint8)
    upper_hsv = np.array([25, 170,255], dtype=np.uint8)
    hsv_mask  = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Combine both masks — pixel must pass BOTH
    combined = cv2.bitwise_and(skin_mask, hsv_mask)

    # Remove hair: hair pixels are very dark (low value in HSV)
    # Create a "not dark" mask
    _, not_dark = cv2.threshold(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY),
        40, 255, cv2.THRESH_BINARY
    )
    combined = cv2.bitwise_and(combined, not_dark)

    # Remove very bright highlights (specular reflections, not skin features)
    _, not_bright = cv2.threshold(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY),
        240, 255, cv2.THRESH_BINARY_INV
    )
    combined = cv2.bitwise_and(combined, not_bright)

    # Morphological cleanup — fill small holes, remove noise
    kernel  = np.ones((7, 7), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)

    # Dilate slightly to ensure edges are included
    combined = cv2.dilate(combined, np.ones((5,5), np.uint8), iterations=1)

    return combined


def apply_skin_mask_to_detection(mask, skin_mask):
    """
    Filter a detection mask to only include pixels
    that are also inside the skin mask.
    """
    return cv2.bitwise_and(mask, skin_mask)


def annotate_face(face_image, severity='Unknown'):
    """
    Annotate skin features on face image.
    Only analyses actual skin regions — ignores hair and background.
    """
    img_rgb = face_image.copy()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # ── Get skin-only mask ─────────────────────────────────
    skin_mask = get_skin_mask(img_rgb)

    found  = {}
    canvas = img_bgr.copy()

    # ── 1. Redness — only on skin ──────────────────────────
    redness_mask = _get_redness_mask(hsv)
    redness_mask = apply_skin_mask_to_detection(redness_mask, skin_mask)
    redness_count = _draw_contour_outlines(
        canvas, redness_mask, COLORS['Redness'],
        min_area=200, max_area=6000, thickness=2)
    if redness_count:
        found['Redness'] = redness_count

    # ── 2. Hyperpigmentation — only on skin ────────────────
    hyper_mask = _get_hyperpigmentation_mask(lab, gray)
    hyper_mask = apply_skin_mask_to_detection(hyper_mask, skin_mask)
    hyper_count = _draw_contour_outlines(
        canvas, hyper_mask, COLORS['Hyperpigmentation'],
        min_area=60, max_area=2000, thickness=2)
    if hyper_count:
        found['Hyperpigmentation'] = hyper_count

    # ── 3. Pustules — only on skin ─────────────────────────
    pustule_mask = _get_pustule_mask(hsv)
    pustule_mask = apply_skin_mask_to_detection(pustule_mask, skin_mask)
    pustule_count = _draw_dotted_contours(
        canvas, pustule_mask, COLORS['Pustule'],
        min_area=30, max_area=500)
    if pustule_count:
        found['Pustules'] = pustule_count

    # ── 4. Papules — only on skin ──────────────────────────
    papule_mask = _get_papule_mask(img_rgb, hsv)
    papule_mask = apply_skin_mask_to_detection(papule_mask, skin_mask)
    papule_count = _draw_contour_outlines(
        canvas, papule_mask, COLORS['Papule'],
        min_area=40, max_area=400, thickness=1)
    if papule_count:
        found['Papules'] = papule_count

    # ── 5. Blackheads — only on skin ──────────────────────
    bh_mask = _get_blackhead_mask(gray)
    bh_mask = apply_skin_mask_to_detection(bh_mask, skin_mask)
    bh_count = _draw_small_circles(
        canvas, bh_mask, COLORS['Blackhead'],
        min_area=3, max_area=80)
    if bh_count:
        found['Blackheads'] = bh_count

    # ── 6. Whiteheads — only on skin ──────────────────────
    wh_mask = _get_whitehead_mask(img_rgb, gray)
    wh_mask = apply_skin_mask_to_detection(wh_mask, skin_mask)
    wh_count = _draw_small_circles(
        canvas, wh_mask, COLORS['Whitehead'],
        min_area=4, max_area=100)
    if wh_count:
        found['Whiteheads'] = wh_count

    # ── 7. Rough texture — only on skin ───────────────────
    tex_mask = _get_rough_texture_mask(gray)
    tex_mask = apply_skin_mask_to_detection(tex_mask, skin_mask)
    tex_count = _draw_texture_overlay(
        canvas, tex_mask, COLORS['Rough Texture'])
    if tex_count:
        found['Rough Texture'] = tex_count

    # ── Draw skin boundary (optional visual) ──────────────
    _draw_skin_boundary(canvas, skin_mask)

    # ── Legend panel ──────────────────────────────────────
    canvas = _draw_legend(canvas, found, severity)

    return canvas, found


# ── Mask generators ────────────────────────────────────────────

def _get_redness_mask(hsv):
    m1   = cv2.inRange(hsv, np.array([0,  50, 70]),  np.array([10, 255,255]))
    m2   = cv2.inRange(hsv, np.array([168,50, 70]),  np.array([180,255,255]))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((12,12),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((6, 6), np.uint8))
    return mask


def _get_hyperpigmentation_mask(lab, gray):
    l, a, b = cv2.split(lab)
    # Adjust thresholds based on overall luminance to reduce false positives on dark skin
    mean_l = int(np.mean(l))
    if mean_l > 140:
        l_thresh = 100
        gray_thresh = 85
    elif mean_l > 100:
        l_thresh = 92
        gray_thresh = 80
    elif mean_l > 70:
        l_thresh = 82
        gray_thresh = 70
    else:
        l_thresh = 72
        gray_thresh = 60

    _, dark  = cv2.threshold(l,    l_thresh,    255, cv2.THRESH_BINARY_INV)
    _, vdark = cv2.threshold(gray, gray_thresh, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_or(dark, vdark)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((8,8),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((4,4),np.uint8))
    return mask


def _get_pustule_mask(hsv):
    m1  = cv2.inRange(hsv, np.array([0,  60, 90]),  np.array([10, 255,255]))
    m2  = cv2.inRange(hsv, np.array([168,60, 90]),  np.array([180,255,255]))
    red = cv2.bitwise_or(m1, m2)
    _, brt = cv2.threshold(hsv[:,:,2], 190, 255, cv2.THRESH_BINARY)
    mask   = cv2.bitwise_and(red, brt)
    mask   = cv2.dilate(mask, np.ones((4,4),np.uint8), iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((6,6),np.uint8))
    return mask


def _get_papule_mask(img_rgb, hsv):
    m1   = cv2.inRange(hsv, np.array([0,  45,55]),  np.array([14, 255,200]))
    m2   = cv2.inRange(hsv, np.array([166,45,55]),  np.array([180,255,200]))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((6,6),np.uint8))
    return mask


def _get_blackhead_mask(gray):
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    mean_g   = int(np.mean(blurred))
    # For darker skin tones the global mean is lower; lower threshold to avoid marking large areas
    if mean_g > 120:
        thr = 60
    elif mean_g > 85:
        thr = 48
    else:
        thr = 38

    _, mask  = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY_INV)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
    return mask


def _get_whitehead_mask(img_rgb, gray):
    _, bright = cv2.threshold(gray, 218, 255, cv2.THRESH_BINARY)
    r, g, b   = cv2.split(img_rgb)
    yellow    = ((r>195) & (g>190) & (b<215)).astype(np.uint8) * 255
    mask      = cv2.bitwise_and(bright, yellow)
    mask      = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
    return mask


def _get_rough_texture_mask(gray):
    lap     = cv2.Laplacian(gray, cv2.CV_64F)
    abs_lap = np.uint8(np.absolute(lap))
    _, mask = cv2.threshold(abs_lap, 18, 255, cv2.THRESH_BINARY)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10,10),np.uint8))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((6, 6), np.uint8))
    return mask


# ── Drawing functions ──────────────────────────────────────────

def _draw_skin_boundary(canvas, skin_mask):
    """Draw a subtle outline showing what was detected as skin."""
    cnts, _ = cv2.findContours(
        skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest, True)
        smooth  = cv2.approxPolyDP(largest, epsilon, True)
        cv2.drawContours(canvas, [smooth], -1, (180,180,180), 1)


def _draw_contour_outlines(canvas, mask, color, min_area, max_area, thickness=2):
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            epsilon = 0.02 * cv2.arcLength(c, True)
            smooth  = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(canvas, [smooth], -1, color, thickness)
            count += 1
    return count


def _draw_dotted_contours(canvas, mask, color, min_area, max_area):
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            epsilon = 0.015 * cv2.arcLength(c, True)
            smooth  = cv2.approxPolyDP(c, epsilon, True)
            pts     = smooth.reshape(-1, 2)
            step    = max(1, len(pts) // 20)
            for i in range(0, len(pts), step):
                cx, cy = pts[i]
                cv2.circle(canvas, (int(cx), int(cy)), 2, color, -1)
            count += 1
    return count


def _draw_small_circles(canvas, mask, color, min_area, max_area):
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            r  = max(3, int(np.sqrt(area / np.pi)))
            cv2.circle(canvas, (cx, cy), r,      color, 1)
            cv2.circle(canvas, (cx, cy), r // 2, color, -1)
            count += 1
    return count


def _draw_texture_overlay(canvas, mask, color):
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count   = 0
    overlay = canvas.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        if 400 < area < 12000:
            cv2.drawContours(overlay, [c], -1, color, -1)
            count += 1
    cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)
    return count


def _draw_legend(canvas, found, severity):
    if not found:
        return canvas

    H, W    = canvas.shape[:2]
    lines   = list(found.keys())
    pad     = 12
    line_h  = 24
    panel_h = pad * 2 + len(lines) * line_h + 34
    panel_w = 220
    # Place legend near the bottom-right corner to avoid covering forehead/eyes
    px      = W - panel_w - 10
    py      = max(10, H - panel_h - 10)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (px, py),
                  (px + panel_w, py + panel_h), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    cv2.rectangle(canvas, (px, py),
                  (px + panel_w, py + panel_h), (70,70,70), 1)

    cv2.putText(canvas, 'Skin Analysis',
                (px + pad, py + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220,220,220), 1)

    row = py + pad + 24
    for feature in lines:
        col = COLORS.get(feature, (200,200,200))
        cv2.circle(canvas, (px + pad + 6, row - 5), 6, col, -1)
        cv2.putText(canvas, feature,
                    (px + pad + 18, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210,210,210), 1)
        row += line_h

    sev_col = (80,255,80)  if severity == 'Clear Skin'    else \
              (80,200,255) if severity == 'Mild Acne'     else \
              (80,140,255) if severity == 'Moderate Acne' else \
              (60,60, 255)
    cv2.putText(canvas, f'Severity: {severity}',
                (px + pad, row + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, sev_col, 1)

    return canvas