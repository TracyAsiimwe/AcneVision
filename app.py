import os
import uuid
import base64
import cv2
from datetime import datetime
import requests
from dotenv import load_dotenv
load_dotenv()


from chatbot.simple_bot import generate_ai_response
from chatbot.simple_bot import warmup_model
warmup_model()

# Try to import medical report generator; provide fallback if not available
try:
    from utils.skin_explainer import generate_full_medical_report
except ImportError:
    def generate_full_medical_report(severity, features, health_score, confidence):
        return f"<p>Medical report for {severity} (confidence: {confidence}%). Health score: {health_score}/100.</p>"

# Try to import evidence explainer; provide fallback if not available
try:
    from utils.skin_explainer import get_evidence_explanation
except ImportError:
    def get_evidence_explanation(raw_class, features, confidence, found_features):
        severity_map = {
            'clear_skin': ('Clear Skin', '#059669', '0 lesions'),
            'mild': ('Mild Acne', '#d97706', '< 20 lesions'),
            'moderate': ('Moderate Acne', '#ea580c', '20–50 lesions'),
            'severe': ('Severe Acne', '#dc2626', '> 50 lesions'),
        }
        sev_display, sev_color, lesion_range = severity_map.get(raw_class, ('Unknown', '#888888', 'N/A'))
        detected = []
        if found_features:
            for name, count in found_features.items():
                detected.append(f"Detected {count} {name.lower()} region(s)")
        if not detected:
            detected.append("No significant lesions detected in the analyzed regions.")
        return {
            'severity_display': sev_display,
            'severity_color': sev_color,
            'lesion_range': lesion_range,
            'clinical_evidence': f"The AI model classified this image as {sev_display.lower()} based on visual pattern analysis.",
            'detected_evidence': detected,
            'location_findings': [
                "Forehead: Analyzed for comedones and inflammation.",
                "Cheeks: Checked for papules and redness.",
                "Nose: Examined for blackhead density.",
                "Chin: Evaluated for acne clusters.",
                "Jawline: Assessed for inflammatory activity."
            ],
            'acne_type_name': 'Comedonal / Inflammatory Acne' if raw_class in ['mild', 'moderate', 'severe'] else 'Clear Skin',
            'acne_type_desc': 'A mix of non-inflammatory and inflammatory lesions commonly triggered by excess sebum and bacterial colonization.',
            'acne_type_cause': 'Hormonal fluctuations, diet, stress, and improper skincare routines.',
            'what_this_means': f'Your skin shows signs of {sev_display.lower()}. Consistent care and appropriate treatment can help improve your skin condition.',
            'what_to_do': [
                'Use a gentle, non-comedogenic cleanser twice daily.',
                'Apply topical treatments containing salicylic acid or benzoyl peroxide.',
                'Avoid picking or squeezing lesions to prevent scarring.',
                'Maintain a consistent skincare routine and stay hydrated.',
                'Consult a dermatologist if condition persists or worsens.'
            ]
        }

from flask import (
    Flask, render_template, request,
    jsonify, redirect, url_for, flash, session,
    Response, stream_with_context
)
from werkzeug.utils import secure_filename
from flask_login import (
    LoginManager, login_required,
    current_user, login_user, logout_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# ── Utils ──────────────────────────────────────────────────
from utils.face_detection import detect_face
from utils.prediction import predict_acne_severity, load_model_once
from utils.gradcam import generate_gradcam_heatmap
from utils.skin_features import analyze_skin_features
from utils.report_generator import generate_skin_report
from utils.gemini_text_generator import generate_gemini_insights

# =========================================================
# AI HELPER FUNCTIONS
# =========================================================

def determine_level(value):
    if value < 15:   return "Minimal"
    elif value < 30: return "Low"
    elif value < 50: return "Moderate"
    elif value < 70: return "Noticeable"
    else:            return "Severe"


def determine_progress(value):
    if value < 15:   return 20
    elif value < 30: return 40
    elif value < 50: return 60
    elif value < 70: return 80
    else:            return 95


def generate_ai_summary(severity, health_score):
    summaries = {
        "clear_skin": (
            f"Your skin appears healthy with balanced texture and minimal acne activity. "
            f"The AI detected very few inflammatory regions across all facial zones, "
            f"suggesting a stable skin condition. "
            f"Your skin health score of {health_score:.0f}/100 reflects a positive result."
        ),
        "mild": (
            f"The analysis detected mild acne activity with small clusters of blemishes "
            f"and minor congestion in certain facial regions. "
            f"Your skin health score is {health_score:.0f}/100. "
            f"Mild acne typically responds well to over-the-counter treatments "
            f"within 6–8 weeks of consistent use."
        ),
        "moderate": (
            f"The AI identified moderate acne patterns including inflammatory bumps, "
            f"redness, and uneven texture across multiple facial zones. "
            f"Your skin health score is {health_score:.0f}/100. "
            f"Moderate acne often benefits from a dermatologist consultation, "
            f"as prescription treatments may be more effective than OTC options."
        ),
        "severe": (
            f"The system detected widespread acne activity with visible inflammation "
            f"and high lesion concentration across several facial regions. "
            f"Your skin health score of {health_score:.0f}/100 reflects significant skin distress. "
            f"Severe acne requires professional medical treatment — "
            f"please consult a dermatologist as soon as possible to prevent scarring."
        ),
    }
    return summaries.get(severity, "Skin analysis complete.")


def generate_face_region_analysis(features):
    blackheads = features.get('blackheads', {}).get('density', 0)
    whiteheads = features.get('whiteheads', {}).get('density', 0)
    papules    = features.get('papules',    {}).get('density', 0)
    redness    = features.get('redness',    {}).get('inflammation_score', 0)
    return {
        "forehead": (
            "Mild acne activity with small whitehead clusters visible."
            if whiteheads > 20
            else "Forehead region appears relatively clear."
        ),
        "cheeks": (
            "Inflammatory papules and redness detected around cheek area."
            if papules > 30 or redness > 30
            else "Cheek region shows minimal irritation."
        ),
        "nose": (
            "Noticeable blackhead concentration and visible pores detected."
            if blackheads > 25
            else "Nose area appears clean with minimal congestion."
        ),
        "chin": (
            "Small acne clusters visible around chin region."
            if papules > 20
            else "Chin area appears balanced and clear."
        ),
        "jawline": (
            "Mild inflammatory activity detected along jawline."
            if redness > 35
            else "Jawline texture appears smooth."
        ),
    }


def generate_condition_insights(features):
    bh = features.get('blackheads',       {}).get('density', 0)
    wh = features.get('whiteheads',       {}).get('density', 0)
    pa = features.get('papules',          {}).get('density', 0)
    rd = features.get('redness',          {}).get('inflammation_score', 0)
    hp = features.get('hyperpigmentation',{}).get('density', 0)
    return {
        "blackheads": (
            "Concentrated pore congestion detected around oil-prone facial zones."
            if bh > 25 else "Only minor blackhead formation detected."
        ),
        "whiteheads": (
            "Small closed-comedone clusters visible across forehead and chin."
            if wh > 25 else "Minimal whitehead activity observed."
        ),
        "papules": (
            "Inflammatory acne bumps detected with localized redness."
            if pa > 30 else "Low papule activity detected."
        ),
        "redness": (
            "Visible skin irritation and inflammatory response identified."
            if rd > 30 else "Skin tone appears relatively balanced."
        ),
        "hyperpigmentation": (
            "Post-acne dark marks detected in several regions."
            if hp > 25 else "Minimal hyperpigmentation visible."
        ),
    }


# =========================================================
# NEW HELPER FUNCTIONS FOR LOVABLE-STYLE RESULTS UI
# =========================================================

def determine_skin_type(features):
    """
    Infer skin type from feature scores.
    Returns: 'Oily', 'Dry', 'Combination', or 'Normal'
    """
    bh = features.get('blackheads', {}).get('density', 0)
    wh = features.get('whiteheads', {}).get('density', 0)
    rd = features.get('redness',    {}).get('inflammation_score', 0)
    tx = features.get('texture_roughness', {}).get('roughness_score', 0)
    pores = features.get('pores', {}).get('visibility', 'Low')

    oily_score = 0
    if bh > 20: oily_score += 2
    if pores == 'High': oily_score += 2
    elif pores == 'Moderate': oily_score += 1
    if wh > 15: oily_score += 1

    dry_score = 0
    if tx > 40: dry_score += 2
    if rd > 25: dry_score += 1
    if pores == 'Low': dry_score += 1

    if oily_score >= 3 and dry_score >= 2:
        return 'Combination'
    elif oily_score >= 3:
        return 'Oily'
    elif dry_score >= 3:
        return 'Dry'
    else:
        return 'Normal'


def determine_inflammation_label(features):
    """
    Returns a human-readable inflammation label.
    """
    rd = features.get('redness', {}).get('inflammation_score', 0)
    pa = features.get('papules', {}).get('density', 0)
    pu = features.get('pustules', {}).get('density', 0)

    total = rd * 0.5 + pa * 0.3 + pu * 0.2
    if total >= 45:   return 'Severe'
    elif total >= 28: return 'Moderate'
    elif total >= 12: return 'Mild'
    else:             return 'Minimal'


def determine_scarring_risk(raw_class, features):
    """
    Returns scarring risk: 'Low', 'Moderate', or 'High'
    """
    pa = features.get('papules',  {}).get('density', 0)
    pu = features.get('pustules', {}).get('density', 0)
    hp = features.get('hyperpigmentation', {}).get('density', 0)

    if raw_class == 'severe':
        return 'High'
    elif raw_class == 'moderate' or pa > 30 or pu > 20:
        return 'High' if hp > 20 else 'Moderate'
    elif raw_class == 'mild' or pa > 15:
        return 'Moderate' if hp > 10 else 'Low'
    else:
        return 'Low'


def determine_action_label(raw_class):
    """
    Returns the recommended action label based on severity.
    """
    actions = {
        'clear_skin': 'Maintain Routine',
        'mild':       'OTC Treatment',
        'moderate':   'See A Dermatologist',
        'severe':     'See A Dermatologist',
    }
    return actions.get(raw_class, 'Consult A Professional')


def generate_what_we_observed(raw_class, features, zones):
    """
    Generates the 'What we observed' narrative paragraph shown below the dermatologist alert.
    """
    bh = features.get('blackheads',        {}).get('density', 0)
    pa = features.get('papules',           {}).get('density', 0)
    rd = features.get('redness',           {}).get('inflammation_score', 0)
    hp = features.get('hyperpigmentation', {}).get('density', 0)
    tx = features.get('texture_roughness', {}).get('roughness_score', 0)

    parts = []

    if raw_class == 'clear_skin':
        return ("The analysis found minimal acne activity across all facial zones. "
                "Skin texture appears even with no significant inflammatory lesions detected. "
                "Your skin barrier looks healthy and well-maintained.")

    if pa > 20 or rd > 25:
        parts.append("a pattern of inflammatory acne primarily on the forehead and mid-face")
    if bh > 20:
        parts.append("visible comedonal congestion around the nose and chin")
    if hp > 15:
        parts.append("a significant presence of post-inflammatory hyperpigmentation (dark spots) where previous acne has healed")
    if tx > 40:
        parts.append("notable surface texture irregularities consistent with active or healing lesions")

    if not parts:
        parts.append("mild acne activity with localized blemishes across several facial regions")

    narrative = "We observed " + ", and ".join(parts) + ". "

    active_zones = []
    if "active" in zones.get('forehead', '').lower() or "acne" in zones.get('forehead', '').lower() or "whitehead" in zones.get('forehead', '').lower():
        active_zones.append("forehead")
    if "papules" in zones.get('cheeks', '').lower() or "redness" in zones.get('cheeks', '').lower():
        active_zones.append("cheeks")
    if "blackhead" in zones.get('nose', '').lower():
        active_zones.append("nose")
    if "acne" in zones.get('chin', '').lower() or "cluster" in zones.get('chin', '').lower():
        active_zones.append("chin")
    if "inflammatory" in zones.get('jawline', '').lower():
        active_zones.append("jawline")

    if active_zones:
        narrative += f"The most affected areas are the {', '.join(active_zones)}."

    return narrative


def generate_what_this_means(raw_class, features):
    """
    Generates the 'What this means' explanation paragraph at the bottom.
    """
    hp = features.get('hyperpigmentation', {}).get('density', 0)
    rd = features.get('redness',           {}).get('inflammation_score', 0)
    pa = features.get('papules',           {}).get('density', 0)

    base = {
        'clear_skin': (
            "Your skin is in a healthy state with minimal active acne. "
            "Continue your current routine focusing on gentle cleansing, moisturising, and daily SPF. "
            "Monitor for any new activity and maintain your protective habits."
        ),
        'mild': (
            "Mild acne is the most manageable stage and typically responds well to consistent over-the-counter care. "
            "Products containing salicylic acid or benzoyl peroxide can help clear existing blemishes. "
            "Establishing a stable routine now prevents progression to moderate acne."
        ),
        'moderate': (
            "The skin is currently reacting to clogged pores with active inflammation. "
            "This stage often requires more targeted treatment than OTC products can provide. "
            "A dermatologist can prescribe topical retinoids or antibiotics to reduce bacterial load and inflammation effectively."
        ),
        'severe': (
            "Severe acne involves widespread inflammation that penetrates deeper skin layers. "
            "Without professional treatment, this stage carries a high risk of permanent scarring. "
            "A dermatologist visit is strongly recommended — oral treatments such as isotretinoin may be considered."
        ),
    }

    text = base.get(raw_class, "Please consult a skincare professional for personalised advice.")

    if hp > 20 and raw_class in ['mild', 'moderate', 'severe']:
        text += (" On deeper skin tones, this often leaves behind dark spots (PIH) "
                 "that can last longer than the acne itself. Daily SPF and brightening agents "
                 "like Vitamin C or Azelaic Acid help fade these marks faster.")

    return text


def build_lesion_breakdown(features, zones, raw_class='clear_skin'):
    """
    Builds a list of lesion breakdown dicts for the results page cards.
    Each dict: { name, description, count, areas, color }
    Matches the Lovable UI 'Lesion breakdown' section.
    """
    if raw_class == 'clear_skin':
        return []

    bh  = features.get('blackheads',        {}).get('count', 0)
    wh  = features.get('whiteheads',        {}).get('count', 0)
    pa  = features.get('papules',           {}).get('count', 0)
    pu  = features.get('pustules',          {}).get('count', 0)
    hp  = features.get('hyperpigmentation', {}).get('count', 0)
    rd  = features.get('redness',           {}).get('inflammation_score', 0)

    breakdown = []

    # Build area strings from zone findings
    forehead_active = "Forehead" if ("acne" in zones.get('forehead','').lower()
                                     or "whitehead" in zones.get('forehead','').lower()
                                     or "active" in zones.get('forehead','').lower()) else None
    cheeks_active   = "Cheeks"   if ("papule" in zones.get('cheeks','').lower()
                                     or "redness" in zones.get('cheeks','').lower()) else None
    nose_active     = "Nose"     if "blackhead" in zones.get('nose','').lower() else None
    chin_active     = "Chin"     if ("cluster" in zones.get('chin','').lower()
                                     or "acne" in zones.get('chin','').lower()) else None
    jaw_active      = "Jawline"  if "inflammatory" in zones.get('jawline','').lower() else None

    # Pustules
    if pu > 0:
        areas = [a for a in [forehead_active, cheeks_active] if a]
        breakdown.append({
            'name': 'Pustules',
            'description': 'Inflamed bumps with a white/yellow tip.',
            'count': pu,
            'areas': ', '.join(areas) if areas else 'Multiple zones',
            'color': '#00b050',
        })

    # Papules
    if pa > 0:
        areas = [a for a in [cheeks_active, chin_active, jaw_active] if a]
        breakdown.append({
            'name': 'Papules',
            'description': 'Small red, tender bumps without pus.',
            'count': pa,
            'areas': ', '.join(areas) if areas else 'Multiple zones',
            'color': '#e74c3c',
        })

    # Blackheads
    if bh > 0:
        areas = [a for a in [nose_active, chin_active] if a]
        breakdown.append({
            'name': 'Blackheads',
            'description': 'Open comedones — oxidized oil in pores.',
            'count': bh,
            'areas': ', '.join(areas) if areas else 'T-zone area',
            'color': '#5c5ce8',
        })

    # Whiteheads
    if wh > 0:
        areas = [a for a in [forehead_active, chin_active] if a]
        breakdown.append({
            'name': 'Whiteheads',
            'description': 'Closed comedones trapped under skin.',
            'count': wh,
            'areas': ', '.join(areas) if areas else 'Forehead, Chin',
            'color': '#00cccc',
        })

    # Post-inflammatory hyperpigmentation
    if hp > 0:
        areas = [a for a in [cheeks_active, jaw_active] if a]
        breakdown.append({
            'name': 'Post-Inflammatory',
            'description': 'Dark marks left after healed acne.',
            'count': hp,
            'areas': ', '.join(areas) if areas else 'Cheeks, Jawline',
            'color': '#f39c12',
        })

    return breakdown


def build_zone_percentages(features, zones, raw_class='clear_skin'):
    """
    Builds zone percentage data for the 'Affected facial zones' bar section.
    Only shows non-zero if there is meaningful activity detected.
    """
    # Clear skin — no zones affected
    if raw_class == 'clear_skin':
        return {'forehead': 0, 'cheeks': 0, 'nose': 0, 'chin': 0, 'jawline': 0}

    bh  = features.get('blackheads',        {}).get('density', 0)
    wh  = features.get('whiteheads',        {}).get('density', 0)
    pa  = features.get('papules',           {}).get('density', 0)
    rd  = features.get('redness',           {}).get('inflammation_score', 0)
    hp  = features.get('hyperpigmentation', {}).get('density', 0)
    pu  = features.get('pustules',          {}).get('density', 0)

    # Forehead: whiteheads + papules + redness
    forehead_raw = (wh * 0.5) + (pa * 0.3) + (rd * 0.2)
    # Cheeks: papules + redness + hyperpigmentation
    cheeks_raw   = (pa * 0.4) + (rd * 0.35) + (hp * 0.25)
    # Nose: blackheads dominant
    nose_raw     = (bh * 0.7) + (pu * 0.3)
    # Chin: blackheads + papules + whiteheads
    chin_raw     = (bh * 0.3) + (pa * 0.4) + (wh * 0.3)
    # Jawline: redness + papules + hyperpigmentation
    jawline_raw  = (rd * 0.5) + (pa * 0.3) + (hp * 0.2)

    # Minimum threshold — below this, zone is not meaningfully affected
    THRESHOLD = 12

    def clamp_pct(val):
        if val < THRESHOLD:
            return 0  # Not meaningfully affected
        return min(100, max(0, int(val)))

    return {
        'forehead': clamp_pct(forehead_raw),
        'cheeks':   clamp_pct(cheeks_raw),
        'nose':     clamp_pct(nose_raw),
        'chin':     clamp_pct(chin_raw),
        'jawline':  clamp_pct(jawline_raw),
    }



# =========================================================
# OBSERVATION & MEANING TEXT GENERATORS
# =========================================================

def generate_observation_text(raw_class, lesions, zones, skin_type):
    """Builds the 'What we observed' narrative paragraph."""
    top_zones = [z.title() for z, p in zones.items() if p >= 20]
    zone_str = " and ".join(top_zones) if top_zones else "across facial zones"

    if raw_class == 'clear_skin':
        return ("The analysis found minimal acne activity across all facial zones. "
                "Skin texture appears even with no significant inflammatory lesions detected. "
                "Your skin barrier looks healthy and well-maintained.")

    parts = []
    if lesions:
        lesion_names = [l['name'].lower() for l in lesions]
        if len(lesion_names) == 1:
            parts.append(f"a pattern of {lesion_names[0]} primarily on the {zone_str}")
        else:
            parts.append(f"a distribution of {', '.join(lesion_names[:-1])} and {lesion_names[-1]} primarily on the {zone_str}")
    else:
        parts.append(f"{raw_class.replace('_', ' ')} skin activity primarily on the {zone_str}")

    if skin_type in ['Oily', 'Combination']:
        parts.append("visible shine in the T-zone (forehead and nose) suggesting active sebum production")

    narrative = "We observed " + ", and ".join(parts) + ". "
    return narrative


def generate_meaning_text(raw_class, skin_type, inflammation):
    """Builds the 'What this means' explanation paragraph."""
    if raw_class == 'clear_skin':
        return ("Your skin is in a healthy state with minimal active acne. "
                "Continue your current routine focusing on gentle cleansing, moisturising, and daily SPF. "
                "Monitor for any new activity and maintain your protective habits.")

    base = {
        'mild': (
            "Mild acne is the most manageable stage and typically responds well to consistent over-the-counter care. "
            "Products containing salicylic acid or benzoyl peroxide can help clear existing blemishes. "
            "Establishing a stable routine now prevents progression to moderate acne."
        ),
        'moderate': (
            "The skin is currently reacting to clogged pores with active inflammation. "
            "This stage often requires more targeted treatment than OTC products can provide. "
            "A dermatologist can prescribe topical retinoids or antibiotics to reduce bacterial load and inflammation effectively."
        ),
        'severe': (
            "Severe acne involves widespread inflammation that penetrates deeper skin layers. "
            "Without professional treatment, this stage carries a high risk of permanent scarring. "
            "A dermatologist visit is strongly recommended — oral treatments such as isotretinoin may be considered."
        ),
    }

    text = base.get(raw_class, "Please consult a skincare professional for personalised advice.")

    if skin_type in ['Oily', 'Combination']:
        text += (f" On {skin_type.lower()} skin types, excess sebum production contributes to pore clogging. "
                 "Incorporating niacinamide and lightweight, non-comedogenic moisturisers helps regulate oil without stripping the barrier.")

    if inflammation in ['Moderate', 'Severe']:
        text += (f" The {inflammation.lower()} inflammation level indicates active immune response. "
                 "Anti-inflammatory ingredients like azelaic acid or centella asiatica can help calm redness while treating the underlying cause.")

    return text


# =========================================================
# FLASK APP SETUP
# =========================================================

app = Flask(__name__)
app.jinja_env.globals['enumerate'] = enumerate
app.secret_key = os.getenv('SECRET_KEY', 'acnevision-dev-key-change-in-prod')

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, 'static', 'uploads')
HEATMAP_FOLDER = os.path.join(BASE_DIR, 'static', 'heatmaps')
REPORT_FOLDER  = os.path.join(BASE_DIR, 'static', 'reports')
DB_FOLDER      = os.path.join(BASE_DIR, 'database')

for folder in [UPLOAD_FOLDER, HEATMAP_FOLDER, REPORT_FOLDER, DB_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER']               = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER']              = HEATMAP_FOLDER
app.config['REPORT_FOLDER']               = REPORT_FOLDER
app.config['MAX_CONTENT_LENGTH']          = 16 * 1024 * 1024
_db_url = os.getenv('DATABASE_URL', f"sqlite:///{os.path.join(DB_FOLDER, 'app.db')}")
if _db_url.startswith('postgres://'):          # Render gives postgres://, SQLAlchemy needs postgresql://
    _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = _db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ── Database ───────────────────────────────────────────────
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy(app)


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id         = db.Column(db.Integer,     primary_key=True)
    username   = db.Column(db.String(80),  unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime,   default=datetime.utcnow)
    analyses   = db.relationship('Analysis', backref='user', lazy=True)


class Analysis(db.Model):
    __tablename__ = 'analyses'
    id                      = db.Column(db.Integer, primary_key=True)
    user_id                 = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    severity                = db.Column(db.String(50),  nullable=False)
    confidence              = db.Column(db.Float,        nullable=False)
    health_score            = db.Column(db.Integer,      nullable=False)
    blackheads_level        = db.Column(db.String(20))
    whiteheads_level        = db.Column(db.String(20))
    papules_level           = db.Column(db.String(20))
    redness_level           = db.Column(db.String(20))
    hyperpigmentation_level = db.Column(db.String(20))
    texture_level           = db.Column(db.String(20))
    face_image              = db.Column(db.String(200))
    annotated_image         = db.Column(db.String(200))
    heatmap_image           = db.Column(db.String(200))
    report_file             = db.Column(db.String(200))
    created_at              = db.Column(db.DateTime, default=datetime.utcnow)


# ── Login manager ──────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view    = 'login'
login_manager.login_message = 'Please sign in to use AcneVision.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Create tables ──────────────────────────────────────────
with app.app_context():
    db.create_all()
    print("[INFO] Database ready.")

# ── Allowed files ──────────────────────────────────────────
ALLOWED = {'png', 'jpg', 'jpeg', 'jfif', 'webp', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

# ── Load model ─────────────────────────────────────────────
print("[INFO] Loading AI model...")
try:
    MODEL = load_model_once()
    print("[INFO] Model loaded.")
except Exception as e:
    print(f"[WARNING] Model not loaded: {e}")
    MODEL = None

# Warm up phi3 so first chat response is not slow
import threading

def _warmup():
    try:
        from chatbot.simple_bot import warmup_model, check_ollama
        if check_ollama():
            print("[INFO] Warming up phi3...")
            warmup_model()
    except Exception as e:
        print(f"[INFO] phi3 warmup skipped: {e}")

threading.Thread(target=_warmup, daemon=True).start()


# =========================================================
# AUTH ROUTES
# =========================================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email',    '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm',  '')

        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('register.html')
        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'error')
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return render_template('register.html')

        user = User(
            username = username,
            email    = email,
            password = generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash(f'Welcome, {username}! Account created.', 'success')
        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()

        if not user or not check_password_hash(user.password, password):
            flash('Incorrect username or password.', 'error')
            return render_template('login.html')

        login_user(user, remember=True)
        flash(f'Welcome back, {user.username}!', 'success')
        return redirect(url_for('index'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been signed out.', 'success')
    return redirect(url_for('login'))


# =========================================================
# HISTORY ROUTES
# =========================================================

@app.route('/history')
@login_required
def history():
    analyses = Analysis.query.filter_by(
        user_id=current_user.id
    ).order_by(Analysis.created_at.desc()).all()
    return render_template('history.html', analyses=analyses)


@app.route('/history/delete/<int:analysis_id>')
@login_required
def delete_analysis(analysis_id):
    a = Analysis.query.get_or_404(analysis_id)
    if a.user_id != current_user.id:
        flash('Not authorised.', 'error')
        return redirect(url_for('history'))
    db.session.delete(a)
    db.session.commit()
    flash('Analysis deleted.', 'success')
    return redirect(url_for('history'))


@app.route('/history/view/<int:analysis_id>')
@login_required
def view_analysis(analysis_id):
    a = Analysis.query.get_or_404(analysis_id)
    if a.user_id != current_user.id:
        flash('Not authorised.', 'error')
        return redirect(url_for('history'))

    raw_class    = a.severity
    severity     = raw_class.replace('_', ' ').title()
    health_score = a.health_score
    score_offset = 440 - (440 * health_score / 100)
    confidence   = a.confidence

    def progress_from_level(level):
        m = {'Minimal': 20, 'Low': 40, 'Moderate': 60, 'Noticeable': 80, 'Severe': 95}
        return m.get(level, 20)

    zones = {'forehead': 'Clear', 'cheeks': 'Clear', 'nose': 'Clear', 'chin': 'Clear', 'jawline': 'Clear'}

    return render_template('results.html',
        severity          = severity,
        raw_class         = raw_class,
        is_combined       = False,
        confidence        = confidence,
        timestamp         = a.created_at.strftime("%B %d, %Y"),
        probs             = {},
        face_image        = a.face_image,
        annotated_image   = a.annotated_image,
        heatmap_image     = a.heatmap_image,
        found_features    = {},
        ai_summary        = generate_ai_summary(raw_class, health_score),
        medical           = f"<p>{generate_ai_summary(raw_class, health_score)}</p>",
        health_score      = health_score,
        score_offset      = score_offset,
        forehead_finding  = zones['forehead'],
        cheeks_finding    = zones['cheeks'],
        nose_finding      = zones['nose'],
        chin_finding      = zones['chin'],
        jawline_finding   = zones['jawline'],
        blackheads_level           = a.blackheads_level,
        blackheads_progress        = progress_from_level(a.blackheads_level),
        blackheads_insight         = f"Blackhead level: {a.blackheads_level}",
        whiteheads_level           = a.whiteheads_level,
        whiteheads_progress        = progress_from_level(a.whiteheads_level),
        whiteheads_insight         = f"Whitehead level: {a.whiteheads_level}",
        papules_level              = a.papules_level,
        papules_progress           = progress_from_level(a.papules_level),
        papules_insight            = f"Papule level: {a.papules_level}",
        redness_level              = a.redness_level,
        redness_progress           = progress_from_level(a.redness_level),
        redness_insight            = f"Redness level: {a.redness_level}",
        hyperpigmentation_level    = a.hyperpigmentation_level,
        hyperpigmentation_progress = progress_from_level(a.hyperpigmentation_level),
        hyperpigmentation_insight  = f"Hyperpigmentation level: {a.hyperpigmentation_level}",
        texture_level              = a.texture_level,
        texture_progress           = progress_from_level(a.texture_level),
        evidence          = get_evidence_explanation(raw_class, {}, confidence, {}),
        features          = {},
        skin_type         = determine_skin_type({}),
        inflammation_label= determine_inflammation_label({}),
        scarring_risk     = determine_scarring_risk(raw_class, {}),
        action_label      = determine_action_label(raw_class),
        what_we_observed  = generate_what_we_observed(raw_class, {}, zones),
        what_this_means   = generate_what_this_means(raw_class, {}),
        recommendations   = [
            {'label': 'Cleansing', 'title': 'Gentle Cleanser',
             'text': 'Use a mild non-comedogenic cleanser twice daily. Avoid harsh scrubs on inflamed areas.'},
            {'label': 'Habit', 'title': 'Hands Off',
             'text': 'Avoid picking or touching your face. This prevents bacterial spread and scarring.'},
            {'label': 'Lifestyle', 'title': 'Hydration',
             'text': 'Drink 8+ glasses of water daily to support skin healing and barrier function.'},
            {'label': 'Protection', 'title': 'SPF Daily',
             'text': 'Apply SPF 30+ every morning. UV exposure worsens dark marks and slows healing.'},
        ],
        lesion_breakdown  = [],
        zone_percentages  = build_zone_percentages({}, zones, raw_class),
        zone_lesions      = {'forehead': ['No specific lesions detected'], 'cheeks': ['No specific lesions detected'], 'nose': ['No specific lesions detected'], 'chin': ['No specific lesions detected'], 'jawline': ['No specific lesions detected']},
        show_find_derm    = raw_class in ['mild', 'moderate', 'severe'],
        report_file       = a.report_file,
    )


# =========================================================
# MAIN ROUTES
# =========================================================

@app.route('/')
@login_required
def index():
    return render_template('index.html')


# =========================================================
# UPLOAD & ANALYSIS PIPELINE
# =========================================================

def _run_pipeline(face_img, original_filename, ts, uid):
    """
    Shared analysis pipeline used by both upload and webcam.
    Returns a dict of all result values.
    """
    face_filename = f"face_{ts}_{uid}.jpg"
    cv2.imwrite(
        os.path.join(UPLOAD_FOLDER, face_filename),
        cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    )

    if MODEL is None:
        raise Exception("Model not loaded. Run training/train.py first.")
    pred          = predict_acne_severity(face_img, MODEL)
    severity      = pred.get('display_class', pred['class'])
    raw_class     = pred['class']
    confidence    = round(pred.get('combined_conf', pred['confidence']) * 100, 1)
    is_combined   = pred.get('is_combined', False)
    probs         = pred['probabilities']

    features     = analyze_skin_features(face_img)

    # Run lesion detection
    try:
        from utils.lesion_detector import annotate_face
        annotated_bgr, found_features = annotate_face(face_img, severity)
        annotated_filename = f"annotated_{ts}_{uid}.jpg"
        cv2.imwrite(
            os.path.join(UPLOAD_FOLDER, annotated_filename),
            annotated_bgr
        )
    except Exception as le:
        print(f"[WARNING] Lesion detection skipped: {le}")
        annotated_filename = None
        found_features     = {}

    evidence = get_evidence_explanation(
        raw_class      = raw_class,
        features       = features,
        confidence     = confidence,
        found_features = found_features,
    )

    health_score = features.get('skin_health_score', 72)
    score_offset = 440 - (440 * health_score / 100)

    heatmap_filename = f"heatmap_{ts}_{uid}.png"
    heatmap_ok = generate_gradcam_heatmap(
        MODEL, face_img, pred['class_index'],
        os.path.join(HEATMAP_FOLDER, heatmap_filename)
    )

    zones      = generate_face_region_analysis(features)
    insights   = generate_condition_insights(features)

    # ── Try Gemini for explanatory text (summary/observed/recommendations) ──
    gemini_insights = generate_gemini_insights(
        raw_class        = raw_class,
        severity_display = severity,
        health_score     = health_score,
        features         = features,
        found_features   = found_features,
        zones            = zones,
    )

    if gemini_insights:
        ai_summary       = gemini_insights["ai_summary"]
        recommendations  = gemini_insights["recommendations"]
    else:
        ai_summary       = generate_ai_summary(raw_class, health_score)
        recommendations  = None

    medical = generate_full_medical_report(
        severity, features, health_score, confidence
    )

    bh = features.get('blackheads',       {}).get('density', 0)
    wh = features.get('whiteheads',       {}).get('density', 0)
    pa = features.get('papules',          {}).get('density', 0)
    rd = features.get('redness',          {}).get('inflammation_score', 0)
    hp = features.get('hyperpigmentation',{}).get('density', 0)
    tx = features.get('texture_roughness',{}).get('roughness_score', 0)

    # ── NEW: Lovable-style derived data ────────────────────
    skin_type         = determine_skin_type(features)
    inflammation_label= determine_inflammation_label(features)
    scarring_risk     = determine_scarring_risk(raw_class, features)
    action_label      = determine_action_label(raw_class)

    # 1. Build lesion data
    lesion_breakdown  = build_lesion_breakdown(features, zones, raw_class)
    zone_percentages  = build_zone_percentages(features, zones, raw_class)
    show_find_derm    = raw_class in ['mild', 'moderate', 'severe']

    # 2. Build zone_lesions from lesion_breakdown
    zone_lesions = {'forehead': [], 'cheeks': [], 'nose': [], 'chin': [], 'jawline': []}
    for lesion in lesion_breakdown:
        areas = lesion.get('areas', '').lower()
        for zone in zone_lesions:
            if zone in areas:
                zone_lesions[zone].append(lesion['name'])
    for zone in zone_lesions:
        if not zone_lesions[zone]:
            zone_lesions[zone] = ['No specific lesions detected']

    # 3. Generate text ONCE, after everything is built
    what_we_observed = generate_observation_text(raw_class, lesion_breakdown, zone_percentages, skin_type)
    what_this_means  = generate_meaning_text(raw_class, skin_type, inflammation_label)

    # 4. Recommendations fallback
    if recommendations is None:
        recommendations = [
            {'label': 'Cleansing', 'title': 'Gentle Cleanser',
             'text': 'Use a mild non-comedogenic cleanser twice daily.'},
            {'label': 'Habit', 'title': 'Hands Off',
             'text': 'Avoid picking or touching your face.'},
            {'label': 'Lifestyle', 'title': 'Hydration',
             'text': 'Drink 8+ glasses of water daily.'},
            {'label': 'Protection', 'title': 'SPF Daily',
             'text': 'Apply SPF 30+ every morning.'},
        ]

    report_data = {
        'timestamp'    : datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        'severity'     : severity,
        'confidence'   : pred['confidence'],
        'probabilities': probs,
        'skin_features': features,
        'heatmap_path' : heatmap_filename if heatmap_ok else None,
        'original_image': original_filename,
        'face_image'   : face_filename,
    }
    report_html     = generate_skin_report(report_data)
    report_filename = f"report_{ts}_{uid}.html"
    with open(os.path.join(REPORT_FOLDER, report_filename),
              'w', encoding='utf-8') as f:
        f.write(report_html)

    if current_user.is_authenticated:
        try:
            analysis = Analysis(
                user_id                 = current_user.id,
                severity                = raw_class,
                confidence              = confidence,
                health_score            = int(health_score),
                blackheads_level        = determine_level(bh),
                whiteheads_level        = determine_level(wh),
                papules_level           = determine_level(pa),
                redness_level           = determine_level(rd),
                hyperpigmentation_level = determine_level(hp),
                texture_level           = determine_level(tx),
                face_image              = face_filename,
                annotated_image         = annotated_filename,
                heatmap_image           = heatmap_filename if heatmap_ok else None,
                report_file             = report_filename,
            )
            db.session.add(analysis)
            db.session.commit()
            print(f"[INFO] Analysis saved to database (id={analysis.id})")
        except Exception as dbe:
            print(f"[WARNING] Database save failed: {dbe}")

    return dict(
        # ── Core prediction ──
        severity          = severity,
        raw_class         = raw_class,
        is_combined       = is_combined,
        confidence        = confidence,
        timestamp         = datetime.now().strftime("%B %d, %Y"),
        probs             = probs,
        # ── Images ──
        original_image    = original_filename,
        face_image        = face_filename,
        annotated_image   = annotated_filename,
        heatmap_image     = heatmap_filename if heatmap_ok else None,
        # ── Lesion detection (from lesion_detector) ──
        found_features    = found_features,
        # ── AI outputs ──
        ai_summary        = ai_summary,
        medical           = medical,
        # ── Health score ──
        health_score      = health_score,
        score_offset      = score_offset,
        # ── Zone text findings ──
        forehead_finding  = zones['forehead'],
        cheeks_finding    = zones['cheeks'],
        nose_finding      = zones['nose'],
        chin_finding      = zones['chin'],
        jawline_finding   = zones['jawline'],
        # ── Per-feature levels/progress/insights ──
        blackheads_level           = determine_level(bh),
        blackheads_progress        = determine_progress(bh),
        blackheads_insight         = insights['blackheads'],
        whiteheads_level           = determine_level(wh),
        whiteheads_progress        = determine_progress(wh),
        whiteheads_insight         = insights['whiteheads'],
        papules_level              = determine_level(pa),
        papules_progress           = determine_progress(pa),
        papules_insight            = insights['papules'],
        redness_level              = determine_level(rd),
        redness_progress           = determine_progress(rd),
        redness_insight            = insights['redness'],
        hyperpigmentation_level    = determine_level(hp),
        hyperpigmentation_progress = determine_progress(hp),
        hyperpigmentation_insight  = insights['hyperpigmentation'],
        texture_level              = determine_level(tx),
        texture_progress           = determine_progress(tx),
        # ── Evidence explainer ──
        evidence                   = evidence,
        # ── Raw features dict (for chatbot context) ──
        features                   = {
            'blackheads'        : features.get('blackheads', {}),
            'whiteheads'        : features.get('whiteheads', {}),
            'papules'           : features.get('papules', {}),
            'pustules'          : features.get('pustules', {}),
            'redness'           : features.get('redness', {}),
            'hyperpigmentation' : features.get('hyperpigmentation', {}),
            'texture_roughness' : features.get('texture_roughness', {}),
        },
        # ── NEW: Lovable-style UI data ──────────────────────
        skin_type         = skin_type,
        inflammation_label= inflammation_label,
        scarring_risk     = scarring_risk,
        action_label      = action_label,
        what_we_observed  = what_we_observed,
        what_this_means   = what_this_means,
        recommendations   = recommendations,
        lesion_breakdown  = lesion_breakdown,
        zone_percentages  = zone_percentages,
        zone_lesions      = zone_lesions,
        show_find_derm    = show_find_derm,
        report_file       = report_filename,
    )


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'image' not in request.files:
        flash("No image uploaded.")
        return redirect(url_for('index'))

    file = request.files['image']

    if file.filename == '':
        flash("No image selected.")
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash("Unsupported format. Use JPG, PNG, WEBP or BMP.")
        return redirect(url_for('index'))

    try:
        uid = uuid.uuid4().hex[:8]
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{ts}_{uid}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"[INFO] Saved: {filepath}")

        face_img, coords = detect_face(filepath)
        if face_img is None:
            return render_template(
                'index.html',
                error="No face detected. Please upload a clear frontal face photo."
            )

        result = _run_pipeline(face_img, filename, ts, uid)
        return render_template('results.html', **result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Analysis failed: {e}")
        return redirect(url_for('index'))


@app.errorhandler(413)
def request_entity_too_large(error):
    flash('Upload failed: file too large. Max size is 16 MB.', 'error')
    return redirect(url_for('index'))


@app.route('/webcam_capture', methods=['POST'])
@login_required
def webcam_capture():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    try:
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        uid = uuid.uuid4().hex[:8]
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"webcam_{ts}_{uid}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, 'wb') as f:
            f.write(img_bytes)

        face_img, coords = detect_face(filepath)
        if face_img is None:
            return jsonify({'error': 'No face detected in webcam photo'}), 400

        result = _run_pipeline(face_img, filename, ts, uid)
        session['result'] = result
        return jsonify({'success': True, 'redirect': url_for('webcam_result')})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/webcam_result')
@login_required
def webcam_result():
    result = session.pop('result', None)
    if not result:
        return redirect(url_for('index'))
    return render_template('results.html', **result)

# =========================================================
# CLINICS PAGE ROUTE
# =========================================================

@app.route('/clinics', methods=['GET'])
@login_required
def clinics():
    local_directory = [
    {
        "name": "Derma Skin Clinic Uganda",
        "type": "Dermatology Clinic",
        "address": "Block C, F1-12, Ntinda Complex, Plot 31 Ntinda - Kisaasi Rd, Kampala",
        "phone": "+256 782 932937",
        "hours": "9:00 AM - 6:00 PM (Mon - Sat), Closed Sunday",
        "rating": "3.8",
        "reviews": "26",
        "lat": 0.3524,
        "lng": 32.6105
    },
    {
        "name": "Unity Skin Clinic",
        "type": "Dermatology Clinic",
        "address": "3rd Floor, Acacia Mall, Cooper Rd, Kisementi, Kololo, Kampala",
        "phone": "+256 786 222457",
        "hours": "8:00 AM - 4:00 PM Daily",
        "rating": "3.8",
        "reviews": "192",
        "lat": 0.3342,
        "lng": 32.5871
    },
    {
        "name": "Glazer Skin Clinic",
        "type": "Dermatology Clinic",
        "address": "1st Floor, Lightning City Building, Opposite TMT Supermarket, Kira Road, Bukoto, Kampala",
        "phone": "+256 760 493634",
        "hours": "8:00 AM - 8:00 PM (Mon - Fri), 8:00 AM - 7:00 PM (Sat), 9:00 AM - 5:00 PM (Sun)",
        "rating": "4.7",
        "reviews": "215",
        "lat": 0.3401,
        "lng": 32.5897
    },
    {
        "name": "Advanced Skin Clinic Uganda",
        "type": "Dermatology Clinic",
        "address": "Susie House, Plot 1001 Ggaba Road, Nsambya, Kampala",
        "phone": "+256 755 562430",
        "hours": "8:30 AM - 8:00 PM (Mon - Sat), 10:00 AM - 5:00 PM (Sun)",
        "rating": "3.9",
        "reviews": "47",
        "lat": 0.3008,
        "lng": 32.6047
    },
    {
        "name": "Kampala Dermatology Clinic",
        "type": "Dermatology Clinic",
        "address": "Plot 37, Bandali Rise, Bugolobi, Kampala",
        "phone": "+256 780 907156",
        "hours": "8:30 AM - 6:00 PM (Mon - Fri), 8:30 AM - 4:00 PM (Sat)",
        "rating": "4.5",
        "reviews": "20",
        "lat": 0.3182,
        "lng": 32.6084
    },
    {
        "name": "Elite Dermatology Ug",
        "type": "Dermatology Clinic",
        "address": "1st Floor Room M05, Mirembe Complex, Kireka, Kampala",
        "phone": "+256 783 676545",
        "hours": "8:00 AM - 6:30 PM (Mon - Sat), Closed Sunday",
        "rating": "4.7",
        "reviews": "57",
        "lat": 0.3470,
        "lng": 32.6492
    },
    {
        "name": "Case Hospital Dermatology Services",
        "type": "Hospital",
        "address": "Plot 4-8, Buganda Road, Kampala",
        "phone": "+256 414 340021",
        "hours": "Open 24 Hours",
        "rating": "4.4",
        "reviews": "140",
        "lat": 0.3156,
        "lng": 32.5727
    },
    {
        "name": "Nakasero Hospital Skin Clinic",
        "type": "Hospital",
        "address": "Plot 14, Akii Bua Road, Nakasero, Kampala",
        "phone": "+256 414 346150",
        "hours": "Open 24 Hours",
        "rating": "3.7",
        "reviews": "202",
        "lat": 0.3190,
        "lng": 32.5822
    },
    {
        "name": "C-Care IHK",
        "type": "Hospital",
        "address": "Plot 4686, Barnabas Road, Namuwongo, Kampala",
        "phone": "+256 312 263300",
        "hours": "Open 24 Hours",
        "rating": "3.8",
        "reviews": "747",
        "lat": 0.3002,
        "lng": 32.6118
    },
    {
        "name": "Mulago National Referral Hospital - Dermatology Clinic",
        "type": "Hospital",
        "address": "Mulago Hill Road, Kampala",
        "phone": "+256 414 554000",
        "hours": "Dermatology OPD on Scheduled Clinic Days",
        "rating": "4.1",
        "reviews": "320",
        "lat": 0.3398,
        "lng": 32.5763
    },
    {
        "name": "UMC Victoria Hospital",
        "type": "Hospital",
        "address": "Plot 1495, Kira Road, Bukoto, Kampala",
        "phone": "+256 312 222555",
        "hours": "Open 24 Hours",
        "rating": "3.6",
        "reviews": "191",
        "lat": 0.3417,
        "lng": 32.5945
    },
    {
        "name": "Rubaga Hospital Skin Clinic",
        "type": "Hospital",
        "address": "Rubaga Road, Kampala",
        "phone": "+256 414 270222",
        "hours": "Open 24 Hours",
        "rating": "4.0",
        "reviews": "115",
        "lat": 0.3015,
        "lng": 32.5526
    }
]
    search_query = request.args.get('query', '').strip().lower()
    active_filter = request.args.get('filter', 'All').strip()

    clean_query = search_query
    if "dermst" in search_query or "dermic" in search_query:
        clean_query = "derm"
    elif "almas" in search_query:
        clean_query = "alma"

    filtered_results = []
    for clinic in local_directory:
        matches_filter = (active_filter == 'All' or clinic['type'] == active_filter)
        matches_query = (
            not clean_query or
            clean_query in clinic['name'].lower() or
            clean_query in clinic['address'].lower() or
            clean_query in clinic['type'].lower()
        )
        if matches_filter and matches_query:
            filtered_results.append(clinic)

    return render_template(
        'clinics.html',
        clinics=filtered_results,
        query=request.args.get('query', ''),
        active_filter=active_filter
    )

@app.route('/learn')
@login_required
def learn():
    return render_template('learn.html')

# =========================================================
# CHATBOT ROUTES
# =========================================================

@app.route('/chatbot/ask', methods=['POST'])
@login_required
def chatbot_ask():
    try:
        from chatbot.simple_bot import generate_ai_response

        data    = request.get_json(silent=True) or {}
        message = str(data.get('message', '')).strip()

        if not message:
            return jsonify({'reply': 'Please type a message first.'})

        severity     = str(data.get('severity',     '') or '').strip()
        health_score = data.get('health_score', None)

        if not severity:
            latest = Analysis.query.filter_by(
                user_id=current_user.id
            ).order_by(Analysis.created_at.desc()).first()
            severity     = latest.severity     if latest else 'clear_skin'
            health_score = latest.health_score if latest else 100

        if health_score is None:
            health_score = 100

        features         = data.get('features',          None)
        found_features   = data.get('found_features',    None)
        location_findings= data.get('location_findings', None)

        reply = generate_ai_response(
            severity         = severity,
            health_score     = int(health_score),
            user_message     = message,
            features         = features,
            found_features   = found_features,
            location_findings= location_findings,
        )

        return jsonify({'reply': str(reply)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'reply': f'Server error: {str(e)}. Please refresh and try again.'
        })


@app.route('/chatbot/stream', methods=['POST'])
@login_required
def chatbot_stream():
    try:
        from chatbot.simple_bot import stream_ai_response

        data    = request.get_json(silent=True) or {}
        message = str(data.get('message', '')).strip()

        if not message:
            return Response("Please type a message first.", content_type='text/plain')

        severity     = str(data.get('severity',     '') or '').strip()
        health_score = data.get('health_score', None)

        if not severity:
            latest = Analysis.query.filter_by(
                user_id=current_user.id
            ).order_by(Analysis.created_at.desc()).first()
            severity     = latest.severity     if latest else 'clear_skin'
            health_score = latest.health_score if latest else 100

        if health_score is None:
            health_score = 100

        features          = data.get('features',          None)
        found_features    = data.get('found_features',    None)
        location_findings = data.get('location_findings', None)

        def generate():
            for token in stream_ai_response(
                severity          = severity,
                health_score      = int(health_score),
                user_message      = message,
                features          = features,
                found_features    = found_features,
                location_findings = location_findings,
            ):
                yield token

        return Response(
            stream_with_context(generate()),
            content_type='text/plain; charset=utf-8',
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(f"Server error: {str(e)}", content_type='text/plain', status=500)


@app.route('/chatbot/status')
@login_required
def chatbot_status():
    try:
        from chatbot.simple_bot import check_ollama, check_model_loaded
        online      = check_ollama()
        model_ready = check_model_loaded() if online else False
        from chatbot.simple_bot import _OLLAMA_UP
        backend = 'ollama' if _OLLAMA_UP else 'gemini'
        return jsonify({
            'online'     : online,
            'model_ready': model_ready,
            'model'      : backend
        })
    except Exception:
        return jsonify({'online': False, 'model_ready': False})


# =========================================================
# RUN
# =========================================================

if __name__ == '__main__':
    print("\n====================================")
    print("  AcneVision")
    print("  Server: http://127.0.0.1:5000")
    print("====================================\n")
    app.run(debug=True, host='0.0.0.0', port=5000)