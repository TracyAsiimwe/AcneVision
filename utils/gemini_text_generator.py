"""
Gemini text generation for AcneVision.
IMPORTANT: This module NEVER classifies severity. It only generates
explanatory text (summary, observations, recommendations) based on
the CNN's existing prediction and feature data.
"""

import os
import json
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def _build_feature_summary(features):
    bh = features.get('blackheads',        {}).get('density', 0)
    wh = features.get('whiteheads',        {}).get('density', 0)
    pa = features.get('papules',           {}).get('density', 0)
    pu = features.get('pustules',          {}).get('density', 0)
    rd = features.get('redness',           {}).get('inflammation_score', 0)
    hp = features.get('hyperpigmentation', {}).get('density', 0)
    tx = features.get('texture_roughness', {}).get('roughness_score', 0)

    return (
        f"Blackheads: {bh:.0f}%, Whiteheads: {wh:.0f}%, Papules: {pa:.0f}%, "
        f"Pustules: {pu:.0f}%, Redness/Inflammation: {rd:.0f}/100, "
        f"Hyperpigmentation: {hp:.0f}%, Texture roughness: {tx:.0f}/100"
    )


def generate_gemini_insights(raw_class, severity_display, health_score,
                              features, found_features, zones):
    """
    Returns a dict with ai_summary, what_we_observed, recommendations,
    or None on any failure (including timeout) so the caller falls back.
    """

    if not GEMINI_API_KEY:
        return None

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        feature_summary = _build_feature_summary(features)

        found_features_str = ", ".join(
            f"{name} ({count} region{'s' if count != 1 else ''})"
            for name, count in (found_features or {}).items()
        ) or "No specific lesions flagged by the detector."

        zone_str = (
            f"Forehead: {zones.get('forehead','')}; "
            f"Cheeks: {zones.get('cheeks','')}; "
            f"Nose: {zones.get('nose','')}; "
            f"Chin: {zones.get('chin','')}; "
            f"Jawline: {zones.get('jawline','')}"
        )

        prompt = f"""
You are a dermatology content writer for a skin analysis app. A CNN model
has ALREADY classified this image. Do NOT change, question, or restate the
classification as your own diagnosis — treat it as fixed fact.

FIXED CLASSIFICATION (from CNN model — do not alter):
- Severity: {severity_display}
- Skin Health Score: {health_score}/100

FEATURE DATA (from image analysis):
- {feature_summary}
- Detected lesions: {found_features_str}
- Facial zone findings: {zone_str}

Generate THREE sections of plain text. Respond ONLY with valid JSON,
no markdown, no code fences, in exactly this structure:

{{
  "ai_summary": "<3-4 sentence professional summary of the overall result, written for the user, referencing the severity level and health score naturally>",
  "what_we_observed": "<2-3 sentence paragraph describing what was observed in the image based on the feature data and zones, written in plain accessible language>",
  "recommendations": [
    {{"label": "Cleansing", "title": "<short title>", "text": "<1-2 sentence actionable recommendation>"}},
    {{"label": "Habit", "title": "<short title>", "text": "<1-2 sentence actionable recommendation>"}},
    {{"label": "Lifestyle", "title": "<short title>", "text": "<1-2 sentence actionable recommendation>"}},
    {{"label": "Protection", "title": "<short title>", "text": "<1-2 sentence actionable recommendation>"}}
  ]
}}

Tailor all four recommendations specifically to the severity level and
detected features above. Never diagnose. Never suggest the severity is
different from what is given. If severity is "Severe Acne", include a
recommendation that strongly encourages seeing a dermatologist.
"""

        # Timeout prevents the request from hanging the Flask worker
        response = model.generate_content(
            prompt,
            request_options={"timeout": 15}
        )
        text = response.text.strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)

        if not all(k in data for k in ("ai_summary", "what_we_observed", "recommendations")):
            return None
        if not isinstance(data["recommendations"], list) or len(data["recommendations"]) == 0:
            return None

        return data

    except Exception as e:
        print(f"[WARNING] Gemini text generation failed: {e}")
        return None