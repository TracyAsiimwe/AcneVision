"""
AcneVision - Prediction with Honest Confidence Ranges
"""
import os
import numpy as np
import cv2
from tensorflow import keras

CLASS_NAMES  = ['clear_skin', 'mild', 'moderate', 'severe']
DISPLAY_NAMES = {
    'clear_skin': 'Clear Skin',
    'mild'      : 'Mild Acne',
    'moderate'  : 'Moderate Acne',
    'severe'    : 'Severe Acne',
}
INPUT_SIZE = (224, 224)

_HERE  = os.path.dirname(os.path.abspath(__file__))
_BASE  = os.path.dirname(_HERE)
_MODEL = None

_SEARCH = [
    os.path.join(_BASE, 'model', 'acne_model.keras'),
    os.path.join(_BASE, 'model', 'acne_model.h5'),
]


def load_model_once():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    for path in _SEARCH:
        if os.path.exists(path):
            print(f"[INFO] Found model at: {path}")
            _MODEL = keras.models.load_model(
                 path,
                 compile=False

            )
            return _MODEL
    raise FileNotFoundError(f"No model found. Searched: {_SEARCH}")


def _get_combined_label(probs_dict):
    """
    If the top two classes are close in probability,
    return a combined label like "Mild to Moderate"
    instead of just "Mild".

    Also defines ADJACENT pairs — only adjacent severity
    levels get combined (not clear_skin + severe).
    """
    adjacent = [
        ('clear_skin', 'mild',     'Clear to Mild'),
        ('mild',       'moderate', 'Mild to Moderate'),
        ('moderate',   'severe',   'Moderate to Severe'),
    ]

    sorted_probs = sorted(
        probs_dict.items(), key=lambda x: x[1], reverse=True
    )

    top1_class, top1_prob = sorted_probs[0]
    top2_class, top2_prob = sorted_probs[1]

    # If top class is confident enough, just return it
    if top1_prob >= 0.65:
        return DISPLAY_NAMES[top1_class], top1_prob, False

    # If top two are close (within 30%), combine them
    if top2_prob > 0.20:
        for a, b, label in adjacent:
            if (top1_class == a and top2_class == b) or \
               (top1_class == b and top2_class == a):
                # Combined confidence = sum of both
                combined_conf = top1_prob + top2_prob
                return label, combined_conf, True

    # Otherwise just return the top class
    return DISPLAY_NAMES[top1_class], top1_prob, False


def predict_acne_severity(face_image, model=None):
    """
    Predict acne severity from a face RGB numpy array.
    Returns a dict with class, display label, confidence,
    and whether it is a combined prediction.
    """
    if model is None:
        model = load_model_once()

    img       = cv2.resize(face_image, INPUT_SIZE)
    img_array = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    raw_probs = model.predict(img_array, verbose=0)[0]

    print("[DEBUG] Probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {raw_probs[i]:.4f}")

    probs_dict = {CLASS_NAMES[i]: float(raw_probs[i])
                  for i in range(len(CLASS_NAMES))}

    top_idx   = int(np.argmax(raw_probs))
    top_class = CLASS_NAMES[top_idx]

    display_label, confidence, is_combined = _get_combined_label(probs_dict)

    return {
        'class'        : top_class,
        'display_class': display_label,   # e.g. "Moderate to Severe"
        'class_index'  : top_idx,
        'confidence'   : float(raw_probs[top_idx]),
        'combined_conf': confidence,
        'is_combined'  : is_combined,
        'probabilities': probs_dict,
    }