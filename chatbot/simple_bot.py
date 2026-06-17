"""
AcneVision Chatbot
- Local development : uses Ollama (llama3.2:3b)
- Production/Render : auto-falls back to Gemini when Ollama is not running
"""
import json
import os
import requests

OLLAMA_URL     = "http://127.0.0.1:11434/api/generate"
OLLAMA_BASE    = "http://127.0.0.1:11434"
MODEL_NAME     = "llama3.2:3b"
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

_OLLAMA_OPTIONS = {
    "temperature": 0.7,
    "num_predict": 120,
    "num_ctx"    : 768,
}

# ── Check Ollama once at startup and cache the result ─────────
def check_ollama():
    try:
        requests.get(OLLAMA_BASE, timeout=3)
        return True
    except Exception:
        return False

_OLLAMA_UP = check_ollama()
print(f"[INFO] Chatbot backend: {'Ollama' if _OLLAMA_UP else 'Gemini'}")


def check_model_loaded():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m.get('name', '') for m in r.json().get('models', [])]
            return any('llama3.2' in m.lower() or 'llama3' in m.lower()
                       for m in models)
        return False
    except Exception:
        return False


def warmup_model():
    if not _OLLAMA_UP:
        return
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": "hi", "stream": False,
                  "options": {"num_predict": 5}},
            timeout=120
        )
        print(f"[INFO] {MODEL_NAME} warmup complete.")
    except Exception as e:
        print(f"[INFO] Warmup skipped: {e}")


# ── Shared prompt builder ─────────────────────────────────────

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


# ── Gemini helpers ────────────────────────────────────────────

def _gemini_response(prompt):
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt, request_options={"timeout": 15})
        return response.text.strip()
    except Exception as e:
        return f"Sorry, I couldn't generate a response right now. Please try again."


def _gemini_stream(prompt):
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield "Sorry, I couldn't generate a response right now. Please try again."


# ── Public API ────────────────────────────────────────────────

def stream_ai_response(severity, health_score, user_message,
                       features=None, found_features=None,
                       location_findings=None):
    """Generator — yields tokens. Uses Ollama locally, Gemini in production."""
    prompt = _build_prompt(severity, health_score, user_message,
                           features, found_features, location_findings)

    if not _OLLAMA_UP:
        if GEMINI_API_KEY:
            yield from _gemini_stream(prompt)
        else:
            yield "Chatbot unavailable. Run Ollama locally or set GEMINI_API_KEY."
        return

    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt,
                  "stream": True, "options": _OLLAMA_OPTIONS},
            stream=True,
            timeout=60,
        )
        for line in r.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    pass
    except requests.exceptions.Timeout:
        yield "The AI is taking too long. Please try again."
    except requests.exceptions.ConnectionError:
        if GEMINI_API_KEY:
            yield from _gemini_stream(prompt)
        else:
            yield "Cannot connect to Ollama. Run: ollama serve"
    except Exception as e:
        yield f"Error: {str(e)}"


def generate_ai_response(severity, health_score, user_message,
                          features=None, found_features=None,
                          location_findings=None):
    """Non-streaming fallback. Uses Ollama locally, Gemini in production."""
    prompt = _build_prompt(severity, health_score, user_message,
                           features, found_features, location_findings)

    if not _OLLAMA_UP:
        if GEMINI_API_KEY:
            return _gemini_response(prompt)
        return "Chatbot unavailable. Run Ollama locally or set GEMINI_API_KEY."

    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt,
                  "stream": False, "options": _OLLAMA_OPTIONS},
            timeout=60,
        )
        if r.status_code == 200:
            content = r.json().get("response", "").strip()
            return content if content else "I received an empty response. Please try again."
        elif r.status_code == 404:
            return f"Model '{MODEL_NAME}' not found. Run: ollama pull {MODEL_NAME}"
        else:
            return f"Ollama error HTTP {r.status_code}. Make sure ollama serve is running."
    except requests.exceptions.Timeout:
        return "The AI is taking too long. Please try again."
    except requests.exceptions.ConnectionError:
        if GEMINI_API_KEY:
            return _gemini_response(prompt)
        return "Cannot connect to Ollama. Run: ollama serve"
    except Exception as e:
        return f"Error: {str(e)}"
