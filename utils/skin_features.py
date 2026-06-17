"""
Skin Feature Analysis - Fixed Version
=======================================
- Only analyses actual skin pixels
- Ignores hair, background, clothing
- More accurate density calculations
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def get_skin_mask(img_rgb):
    """
    Isolate skin pixels using YCrCb + HSV.
    Returns binary mask where 255 = skin pixel.
    """
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    skin1 = cv2.inRange(ycrcb,
                        np.array([0,  133, 77],  dtype=np.uint8),
                        np.array([255,173, 127], dtype=np.uint8))

    hsv   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    skin2 = cv2.inRange(hsv,
                        np.array([0,  15, 50],  dtype=np.uint8),
                        np.array([25, 170,255], dtype=np.uint8))

    combined = cv2.bitwise_and(skin1, skin2)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Remove very dark pixels (hair)
    _, not_dark = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    combined    = cv2.bitwise_and(combined, not_dark)

    # Remove very bright pixels (flash/highlights)
    _, not_bright = cv2.threshold(gray, 242, 255, cv2.THRESH_BINARY_INV)
    combined      = cv2.bitwise_and(combined, not_bright)

    # Clean up
    k        = np.ones((9,9), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k)

    return combined


def analyze_skin_features(face_image):
    """
    Analyze skin features on actual skin pixels only.
    """
    img_rgb = face_image.copy()

    # Get skin mask first
    skin_mask = get_skin_mask(img_rgb)
    skin_area = max(1, np.sum(skin_mask > 0))  # actual skin pixel count

    # If very little skin detected, use full image
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
    skin_ratio   = skin_area / total_pixels
    if skin_ratio < 0.1:
        # Less than 10% skin detected — use full image
        skin_mask = np.ones(img_rgb.shape[:2], dtype=np.uint8) * 255
        skin_area = total_pixels

    # Apply skin mask to image — set non-skin to gray
    skin_only = img_rgb.copy()
    skin_only[skin_mask == 0] = [128, 128, 128]

    lab  = cv2.cvtColor(skin_only, cv2.COLOR_RGB2LAB)
    hsv  = cv2.cvtColor(skin_only, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(skin_only, cv2.COLOR_RGB2GRAY)

    features = {
        'blackheads'       : _detect_blackheads(gray, skin_mask, skin_area),
        'whiteheads'       : _detect_whiteheads(skin_only, gray, skin_mask, skin_area),
        'pustules'         : _detect_pustules(hsv, skin_mask, skin_area),
        'papules'          : _detect_papules(skin_only, hsv, skin_mask, skin_area),
        'redness'          : _analyze_redness(skin_only, hsv, skin_mask, skin_area),
        'hyperpigmentation': _detect_hyperpigmentation(skin_only, lab, skin_mask, skin_area),
        'texture_roughness': _analyze_texture(gray, skin_mask),
        'pores'            : _estimate_pores(gray, skin_mask),
    }
    features['skin_health_score'] = _health_score(features)
    return features


def _detect_skin_tone(img_rgb):
    """
    Detect average skin tone brightness to calibrate thresholds.
    Returns: 'light', 'medium', 'dark', or 'deep'
    """
    # Convert RGB to HSV via BGR path (OpenCV uses BGR internally)
    img_hsv = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    h, w = img_rgb.shape[:2]
    cx, cy = w // 2, h // 2
    # sample a center patch; fallback to whole V channel if small
    y0, y1 = max(0, cy-30), min(h, cy+30)
    x0, x1 = max(0, cx-30), min(w, cx+30)
    sample = img_hsv[y0:y1, x0:x1, 2]
    mean_v = float(np.mean(sample)) if sample.size > 0 else 128.0

    if mean_v > 160:
        return 'light'
    elif mean_v > 110:
        return 'medium'
    elif mean_v > 65:
        return 'dark'
    else:
        return 'deep'


def _detect_blackheads(gray, skin_mask, skin_area):
    """Blackheads: small very dark circular spots on skin."""
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    _, mask  = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    mask     = cv2.bitwise_and(mask, skin_mask)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count   = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if 3 < area < 90:
            peri = cv2.arcLength(c, True)
            if peri > 0 and (4 * np.pi * area / peri**2) > 0.4:
                count += 1

    density = min(100, (count / (skin_area / 1000)) * 100)
    return {
        'count'   : count,
        'density' : round(density, 1),
        'severity': 'High' if density > 25 else 'Moderate' if density > 12 else 'Low',
    }


def _detect_whiteheads(img, gray, skin_mask, skin_area):
    """Whiteheads: small bright yellowish bumps on skin."""
    _, bright = cv2.threshold(gray, 212, 255, cv2.THRESH_BINARY)
    r, g, b   = cv2.split(img)
    yellow    = ((r>190) & (g>185) & (b<212)).astype(np.uint8) * 255
    mask      = cv2.bitwise_and(bright, yellow)
    mask      = cv2.bitwise_and(mask, skin_mask)
    mask      = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count   = sum(1 for c in cnts if 4 < cv2.contourArea(c) < 100)

    density = min(100, (count / (skin_area / 1000)) * 100)
    return {
        'count'   : count,
        'density' : round(density, 1),
        'severity': 'High' if density > 20 else 'Moderate' if density > 10 else 'Low',
    }


def _detect_pustules(hsv, skin_mask, skin_area):
    """Pustules: red + bright center on skin."""
    m1  = cv2.inRange(hsv, np.array([0,  60, 90]),  np.array([10, 255,255]))
    m2  = cv2.inRange(hsv, np.array([168,60, 90]),  np.array([180,255,255]))
    red = cv2.bitwise_or(m1, m2)
    _, brt = cv2.threshold(hsv[:,:,2], 190, 255, cv2.THRESH_BINARY)
    mask   = cv2.bitwise_and(cv2.bitwise_and(red, brt), skin_mask)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count   = sum(1 for c in cnts if 20 < cv2.contourArea(c) < 600)

    density = min(100, (count / (skin_area / 500)) * 100)
    return {
        'count'   : count,
        'density' : round(density, 1),
        'severity': 'High' if density > 15 else 'Moderate' if density > 6 else 'Low',
    }


def _detect_papules(img, hsv, skin_mask, skin_area):
    """Papules: red bumps (no bright center) on skin."""
    m1   = cv2.inRange(hsv, np.array([0,  45,55]),  np.array([14, 255,200]))
    m2   = cv2.inRange(hsv, np.array([166,45,55]),  np.array([180,255,200]))
    mask = cv2.bitwise_and(cv2.bitwise_or(m1, m2), skin_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count   = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if 30 < area < 400:
            x,y,w,h = cv2.boundingRect(c)
            if np.mean(img[y:y+h, x:x+w]) < 185:
                count += 1

    density = min(100, (count / (skin_area / 500)) * 100)
    return {
        'count'   : count,
        'density' : round(density, 1),
        'severity': 'High' if density > 20 else 'Moderate' if density > 8 else 'Low',
    }


def _analyze_redness(img, hsv, skin_mask, skin_area):
    """Redness/inflammation on skin only."""
    m1   = cv2.inRange(hsv, np.array([0,  35,55]),  np.array([14, 255,255]))
    m2   = cv2.inRange(hsv, np.array([166,35,55]),  np.array([180,255,255]))
    mask = cv2.bitwise_and(cv2.bitwise_or(m1, m2), skin_mask)

    red_pixels = np.sum(mask > 0)
    pct        = (red_pixels / skin_area) * 100

    sev   = ('Severe'   if pct > 20 else
             'Moderate' if pct > 10 else
             'Mild'     if pct > 4  else 'Minimal')
    score = 90 if sev=='Severe' else 60 if sev=='Moderate' else 30 if sev=='Mild' else 10

    # Apply skin-tone correction so darker skin does not produce inflated inflammation scores
    try:
        skin_tone = _detect_skin_tone(img)
    except Exception:
        skin_tone = 'medium'

    tone_correction = {
        'light' : 1.0,
        'medium': 0.85,
        'dark'  : 0.65,
        'deep'  : 0.50,
    }
    correction = tone_correction.get(skin_tone, 0.85)
    adj_score = int(round(score * correction))

    r   = img[:,:,0].astype(float)
    gb  = (img[:,:,1].astype(float) + img[:,:,2].astype(float)) / 2
    dom = float(np.mean(r[skin_mask>0]) - np.mean(gb[skin_mask>0])) if skin_area > 0 else 0

    return {
        'redness_percentage': round(pct, 2),
        'severity'          : sev,
        'inflammation_score': adj_score,
        'red_dominance'     : round(dom, 2),
    }


def _detect_hyperpigmentation(img, lab, skin_mask, skin_area):
    """Dark spots on skin."""
    l, a, b = cv2.split(lab)
    _, dark  = cv2.threshold(l,    88, 255, cv2.THRESH_BINARY_INV)
    mask     = cv2.bitwise_and(dark, skin_mask)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((8,8),np.uint8))
    mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((4,4),np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count   = sum(1 for c in cnts if 20 < cv2.contourArea(c) < 2000)

    density = min(100, (count / (skin_area / 800)) * 100)
    return {
        'count'   : count,
        'density' : round(density, 1),
        'severity': 'High' if density > 15 else 'Moderate' if density > 6 else 'Low',
    }


def _analyze_texture(gray, skin_mask):
    """Texture roughness on skin pixels only."""
    skin_gray = cv2.bitwise_and(gray, skin_mask)
    small     = cv2.resize(skin_gray, (128, 128))

    glcm = graycomatrix(small, [1,2,3],
                        [0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    contrast    = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    roughness   = min(100, (contrast/100)*50 + (1-homogeneity)*50)

    return {
        'roughness_score': round(roughness, 1),
        'contrast'       : round(float(contrast), 3),
        'homogeneity'    : round(float(homogeneity), 3),
        'smoothness'     : ('Rough'    if roughness > 60 else
                            'Moderate' if roughness > 35 else 'Smooth'),
    }


def _estimate_pores(gray, skin_mask):
    """Pore visibility on skin only."""
    skin_gray = cv2.bitwise_and(gray, skin_mask)
    blurred   = cv2.GaussianBlur(skin_gray, (5,5), 0)
    edges     = cv2.Canny(blurred, 50, 150)
    edges     = cv2.bitwise_and(edges, skin_mask)

    skin_area = max(1, np.sum(skin_mask > 0))
    density   = np.sum(edges > 0) / skin_area * 100
    vis       = 'High' if density > 8 else 'Moderate' if density > 5 else 'Low'

    return {'visibility': vis, 'edge_density': round(float(density), 2)}


def _health_score(f):
    score = 100
    score -= {'High':15,'Moderate':8,'Low':3}.get(f['blackheads']['severity'], 0)
    score -= {'High':12,'Moderate':6,'Low':2}.get(f['whiteheads']['severity'], 0)
    score -= {'High':20,'Moderate':10,'Low':4}.get(f['pustules']['severity'], 0)
    score -= {'High':18,'Moderate':9,'Low':3}.get(f['papules']['severity'], 0)
    score -= {'Severe':25,'Moderate':15,'Mild':7,'Minimal':0}.get(
              f['redness']['severity'], 0)
    score -= {'High':15,'Moderate':8,'Low':3}.get(
              f['hyperpigmentation']['severity'], 0)
    score -= {'Rough':10,'Moderate':5,'Smooth':0}.get(
              f['texture_roughness']['smoothness'], 0)
    score -= {'High':8,'Moderate':4,'Low':0}.get(f['pores']['visibility'], 0)
    return max(0, min(100, score))