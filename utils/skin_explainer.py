"""
utils/skin_explainer.py
Evidence-based explanation generator for AcneVision.
"""

def generate_full_medical_report(severity, features, health_score, confidence):
    """Generate a full HTML medical report."""
    return f"""
    <div style="font-family: Arial, sans-serif; padding: 20px;">
        <h2>Medical Report</h2>
        <p><strong>Severity:</strong> {severity}</p>
        <p><strong>Confidence:</strong> {confidence}%</p>
        <p><strong>Health Score:</strong> {health_score}/100</p>
    </div>
    """

def get_evidence_explanation(raw_class, features, confidence, found_features):
    """
    Generate an evidence-based explanation for the acne severity prediction.

    Args:
        raw_class: The raw severity class (e.g., 'clear_skin', 'mild', 'moderate', 'severe')
        features: Dict of skin features from analyze_skin_features()
        confidence: Model confidence percentage
        found_features: Dict of detected features with counts

    Returns:
        Dict with all evidence data for the results template.
    """

    # Severity mapping
    severity_data = {
        'clear_skin': {
            'display': 'Clear Skin',
            'color': '#059669',
            'lesion_range': '0 lesions',
            'clinical': 'No active acne lesions detected. Skin appears healthy with minimal inflammation.',
            'type_name': 'No Acne',
            'type_desc': 'Skin is clear with no visible comedones, papules, or pustules.',
            'type_cause': 'N/A — maintain current skincare routine.',
            'meaning': 'Your skin is in excellent condition. Continue your current regimen to maintain results.',
            'actions': [
                'Continue your current gentle skincare routine.',
                'Use SPF 30+ daily to prevent sun damage.',
                'Stay hydrated and maintain a balanced diet.',
                'Cleanse face twice daily with a mild cleanser.'
            ]
        },
        'mild': {
            'display': 'Mild Acne',
            'color': '#d97706',
            'lesion_range': '< 20 lesions',
            'clinical': 'Few comedones and occasional papules. Inflammation is minimal and localized.',
            'type_name': 'Mild Comedonal / Papular Acne',
            'type_desc': 'Characterized by small blackheads, whiteheads, and a few red bumps.',
            'type_cause': 'Excess sebum production, clogged pores, and bacterial growth.',
            'meaning': 'Your acne is mild and typically responds well to over-the-counter treatments within 6–8 weeks.',
            'actions': [
                'Use a salicylic acid or benzoyl peroxide cleanser.',
                'Apply non-comedogenic moisturizer daily.',
                'Avoid touching or picking at your face.',
                'Consider topical retinoids for prevention.',
                'Maintain consistent sleep and hydration habits.'
            ]
        },
        'moderate': {
            'display': 'Moderate Acne',
            'color': '#ea580c',
            'lesion_range': '20–50 lesions',
            'clinical': 'Multiple inflammatory papules and pustules across several facial zones.',
            'type_name': 'Moderate Inflammatory Acne',
            'type_desc': 'Widespread red bumps, pustules, and some nodules. Inflammation is noticeable.',
            'type_cause': 'Increased sebum, Cutibacterium acnes overgrowth, and follicular hyperkeratinization.',
            'meaning': 'Moderate acne often benefits from prescription treatments. A dermatologist consultation is recommended.',
            'actions': [
                'Schedule a dermatologist appointment for prescription options.',
                'Use topical antibiotics or combination therapy as prescribed.',
                'Avoid harsh scrubs that can worsen inflammation.',
                'Consider oral medications if topical treatments are insufficient.',
                'Track triggers (diet, stress, products) in a journal.'
            ]
        },
        'severe': {
            'display': 'Severe Acne',
            'color': '#dc2626',
            'lesion_range': '> 50 lesions',
            'clinical': 'Numerous deep nodules and cysts with significant inflammation and risk of scarring.',
            'type_name': 'Severe Nodulocystic Acne',
            'type_desc': 'Deep, painful cysts and nodules covering large areas. High risk of permanent scarring.',
            'type_cause': 'Severe inflammation, bacterial infection, and genetic predisposition.',
            'meaning': 'Severe acne requires professional medical treatment. Please consult a dermatologist as soon as possible.',
            'actions': [
                'Seek immediate dermatologist consultation.',
                'Discuss isotretinoin or other systemic treatments.',
                'Do not pick or squeeze lesions — scarring risk is high.',
                'Follow a gentle, prescribed skincare regimen only.',
                'Consider mental health support if acne affects wellbeing.'
            ]
        }
    }

    data = severity_data.get(raw_class, severity_data['clear_skin'])

    # Build detected evidence list
    detected = []
    if found_features:
        for name, count in found_features.items():
            detected.append(f"Detected {count} {name.lower()} region(s) in the analyzed facial zones.")
    if not detected:
        detected.append("No significant lesions were detected in the analyzed regions.")

    # Build location findings from features
    bh = features.get('blackheads', {}).get('density', 0)
    wh = features.get('whiteheads', {}).get('density', 0)
    pa = features.get('papules', {}).get('density', 0)
    rd = features.get('redness', {}).get('inflammation_score', 0)
    hp = features.get('hyperpigmentation', {}).get('density', 0)

    locations = []
    if bh > 20:
        locations.append("Nose and T-zone show concentrated blackhead activity due to excess oil production.")
    if wh > 20:
        locations.append("Forehead and chin display closed comedone clusters.")
    if pa > 25:
        locations.append("Cheeks and jawline exhibit inflammatory papule formation.")
    if rd > 25:
        locations.append("Visible redness and inflammation detected across multiple facial zones.")
    if hp > 20:
        locations.append("Post-inflammatory hyperpigmentation marks present on cheeks and jawline.")
    if not locations:
        locations.append("All facial zones appear clear with minimal irregularities.")

    return {
        'severity_display': data['display'],
        'severity_color': data['color'],
        'lesion_range': data['lesion_range'],
        'clinical_evidence': data['clinical'],
        'detected_evidence': detected,
        'location_findings': locations,
        'acne_type_name': data['type_name'],
        'acne_type_desc': data['type_desc'],
        'acne_type_cause': data['type_cause'],
        'what_this_means': data['meaning'],
        'what_to_do': data['actions']
    }