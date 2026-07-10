"""
AcneVision Chatbot — powered by Google Gemini API
"""
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = (
    "You are AcneVision Skin Assistant — a warm, knowledgeable dermatology assistant. "
    "Answer in 2-4 short sentences, conversationally. "
    "If the user's own skin analysis results are provided below, use them to personalize your answer. "
    "If the user asks a general question about acne, skincare, ingredients, routines, or skin "
    "conditions that is NOT about their specific results, answer it normally using your general "
    "knowledge — you are not limited to only discussing their scan. "
    "Never diagnose medical conditions. Recommend a dermatologist for serious or persistent concerns. "
    "Stay focused on skin, acne, dermatology, and skincare topics."
)


def _build_prompt(severity, health_score, user_message,
                  features=None, found_features=None, location_findings=None):
    context_lines = [
        f"Severity: {severity}",
        f"Skin Health Score: {health_score}/100",
    ]

    if features:
        bh = features.get('blackheads',        {}).get('density', 0)
        wh = features.get('whiteheads',        {}).get('density', 0)
        pa = features.get('papules',           {}).get('density', 0)
        pu = features.get('pustules',          {}).get('density', 0)
        rd = features.get('redness',           {}).get('inflammation_score', 0)
        hp = features.get('hyperpigmentation', {}).get('density', 0)
        tx = features.get('texture_roughness', {}).get('roughness_score', 0)
        context_lines.append(
            f"Feature densities — Blackheads: {bh:.0f}%, Whiteheads: {wh:.0f}%, "
            f"Papules: {pa:.0f}%, Pustules: {pu:.0f}%, "
            f"Redness/Inflammation: {rd:.0f}/100, "
            f"Hyperpigmentation: {hp:.0f}%, Texture roughness: {tx:.0f}/100"
        )

    if found_features:
        feat_list = ", ".join(
            f"{name} ({count} region{'s' if count != 1 else ''})"
            for name, count in found_features.items()
        )
        context_lines.append(f"Detected on image: {feat_list}")

    if location_findings:
        context_lines.append("Location findings:")
        for loc in location_findings[:4]:
            context_lines.append(f"  - {loc}")

    context_block = "\n".join(context_lines)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== USER'S SKIN ANALYSIS (for reference, only use if relevant to the question) ===\n"
        f"{context_block}\n\n"
        f"=== USER QUESTION ===\n"
        f"{user_message}\n\n"
        f"Reply:"
    )


def generate_ai_response(severity, health_score, user_message,
                          features=None, found_features=None,
                          location_findings=None):
    """Non-streaming response via Gemini."""
    if not GEMINI_API_KEY:
        return "Chatbot unavailable. Please set GEMINI_API_KEY in your .env file."
    prompt = _build_prompt(severity, health_score, user_message,
                           features, found_features, location_findings)
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt, request_options={"timeout": 15})
        return response.text.strip()
    except Exception:
        return "Sorry, I couldn't generate a response right now. Please try again."


def stream_ai_response(severity, health_score, user_message,
                       features=None, found_features=None,
                       location_findings=None):
    """Streaming token generator via Gemini."""
    if not GEMINI_API_KEY:
        yield "Chatbot unavailable. Please set GEMINI_API_KEY in your .env file."
        return
    prompt = _build_prompt(severity, health_score, user_message,
                           features, found_features, location_findings)
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception:
        yield "Sorry, I couldn't generate a response right now. Please try again."
