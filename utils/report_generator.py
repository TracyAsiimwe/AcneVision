"""
AcneVision Medical Report Generator — Professional Edition
Location: AcneVision/utils/report_generator.py

Generates a dermatologist-ready HTML report that is clear,
clinical, and structured for professional review.
"""

from datetime import datetime


SEVERITY_INFO = {
    'clear_skin': {
        'display'       : 'Clear Skin',
        'icd_code'      : 'L70.0 (Not applicable)',
        'color'         : '#059669',
        'bg'            : '#f0fdf4',
        'border'        : '#86efac',
        'grade'         : 'Grade 0 — No Active Acne',
        'gea_scale'     : '0 / 5',
        'clinical'      : 'No significant acne lesions detected. Skin appears healthy with balanced sebum production, no visible comedones, and no inflammatory markers.',
        'urgency'       : 'No treatment required',
        'urgency_bg'    : '#dcfce7',
        'urgency_color' : '#059669',
        'prognosis'     : 'Excellent. Maintain current skincare routine.',
    },
    'mild': {
        'display'       : 'Mild Acne',
        'icd_code'      : 'L70.0 — Acne vulgaris',
        'color'         : '#d97706',
        'bg'            : '#fffbeb',
        'border'        : '#fcd34d',
        'grade'         : 'Grade I — Mild',
        'gea_scale'     : '1–2 / 5',
        'clinical'      : 'Predominantly non-inflammatory acne with comedones (open and closed) and occasional small papules. Estimated lesion count: 10–30 total lesions confined to one or two facial regions.',
        'urgency'       : 'Over-the-counter treatment recommended',
        'urgency_bg'    : '#fef9c3',
        'urgency_color' : '#a16207',
        'prognosis'     : 'Good. Typically resolves within 6–8 weeks with consistent OTC care.',
    },
    'moderate': {
        'display'       : 'Moderate Acne',
        'icd_code'      : 'L70.0 — Acne vulgaris',
        'color'         : '#ea580c',
        'bg'            : '#fff7ed',
        'border'        : '#fdba74',
        'grade'         : 'Grade II–III — Moderate',
        'gea_scale'     : '3 / 5',
        'clinical'      : 'Mixed inflammatory and non-inflammatory acne. Multiple papules and pustules distributed across several facial zones. Estimated lesion count: 30–125. Early post-inflammatory hyperpigmentation may be present.',
        'urgency'       : 'Dermatologist consultation advised',
        'urgency_bg'    : '#ffedd5',
        'urgency_color' : '#c2410c',
        'prognosis'     : 'Moderate. Prescription-strength topicals or oral therapy significantly improve outcome.',
    },
    'severe': {
        'display'       : 'Severe Acne',
        'icd_code'      : 'L70.0 — Acne vulgaris (severe)',
        'color'         : '#dc2626',
        'bg'            : '#fef2f2',
        'border'        : '#fca5a5',
        'grade'         : 'Grade IV — Severe',
        'gea_scale'     : '4–5 / 5',
        'clinical'      : 'Widespread inflammatory acne with nodules, cysts, and deep-tissue involvement across multiple facial regions. Estimated lesion count: 125+. High risk of permanent scarring without prompt medical intervention.',
        'urgency'       : 'Immediate dermatologist referral strongly recommended',
        'urgency_bg'    : '#fee2e2',
        'urgency_color' : '#b91c1c',
        'prognosis'     : 'Guarded without treatment. Oral isotretinoin or combination therapy may be indicated.',
    },
}

DISPLAY_TO_RAW = {
    'Clear Skin'        : 'clear_skin',
    'Mild Acne'         : 'mild',
    'Moderate Acne'     : 'moderate',
    'Severe Acne'       : 'severe',
    'Clear To Mild'     : 'mild',
    'Mild To Moderate'  : 'moderate',
    'Moderate To Severe': 'severe',
    'Clear_Skin'        : 'clear_skin',
}


def _get_sev_info(severity_label):
    raw = DISPLAY_TO_RAW.get(
        severity_label,
        severity_label.lower().replace(' ', '_').replace('_acne', '')
    )
    return SEVERITY_INFO.get(raw, SEVERITY_INFO['mild']), raw


def _level(v):
    if v < 15:   return 'Minimal',   '#059669', 12
    elif v < 30: return 'Low',       '#2563eb', 28
    elif v < 50: return 'Moderate',  '#d97706', 50
    elif v < 70: return 'Noticeable','#ea580c', 72
    else:        return 'Severe',    '#dc2626', 92


def _feature_row(name, value, unit, level_label, color, bar_w, note):
    return f"""
    <tr style="border-bottom:1px solid #e8e0d5;">
      <td style="padding:11px 16px; font-size:0.85em; font-weight:600; color:#1a1a1a; white-space:nowrap;">
        {name}
      </td>
      <td style="padding:11px 16px; font-size:0.85em; color:#444; text-align:center;">
        {value:.0f}{unit}
      </td>
      <td style="padding:11px 16px; min-width:130px;">
        <div style="background:#ede7dd; border-radius:4px; height:7px; overflow:hidden;">
          <div style="width:{bar_w}%; height:100%; background:{color}; border-radius:4px;"></div>
        </div>
      </td>
      <td style="padding:11px 16px; text-align:center;">
        <span style="display:inline-block; padding:2px 10px; border-radius:10px;
                     font-size:0.76em; font-weight:700; color:{color};
                     background:{color}18; border:1px solid {color}40;">
          {level_label}
        </span>
      </td>
      <td style="padding:11px 16px; font-size:0.82em; color:#5a5a5a; line-height:1.5; max-width:240px;">
        {note}
      </td>
    </tr>"""


def _section_title(title, subtitle=''):
    sub_html = f'<div style="font-size:0.8em;color:#8a8a8a;margin-top:2px;">{subtitle}</div>' if subtitle else ''
    return f"""
    <div style="margin-bottom:14px;">
      <div style="font-family:Georgia,serif; font-size:1.2em; color:#1e2d3d; font-weight:600;">{title}</div>
      {sub_html}
    </div>"""


def generate_skin_report(data):
    now        = datetime.now()
    timestamp  = data.get('timestamp', now.strftime("%B %d, %Y at %I:%M %p"))
    severity   = data.get('severity', 'Mild Acne')
    confidence = data.get('confidence', 0)
    probs      = data.get('probabilities', {})
    features   = data.get('skin_features', {})
    face_img   = data.get('face_image', '')

    conf_pct = round(confidence * 100, 1) if confidence <= 1 else round(confidence, 1)
    sev, raw_class = _get_sev_info(severity)
    report_id = now.strftime("AV-%Y%m%d-%H%M%S")
    date_str  = now.strftime("%d %B %Y")

    def fv(key, sub):
        return features.get(key, {}).get(sub, 0)

    bh = fv('blackheads',        'density')
    wh = fv('whiteheads',        'density')
    pa = fv('papules',           'density')
    pu = fv('pustules',          'density')
    rd = fv('redness',           'inflammation_score')
    hp = fv('hyperpigmentation', 'density')
    tx = fv('texture_roughness', 'roughness_score')
    hs = features.get('skin_health_score', 72)

    # ── Feature rows ─────────────────────────────────────
    feature_data = [
        ('Blackheads (open comedones)', bh, '%',
         'Oxidised sebum plugging open follicles — indicator of pore congestion.',
         'bh_count', fv('blackheads', 'count')),
        ('Whiteheads (closed comedones)', wh, '%',
         'Trapped sebum beneath skin surface — precursors to inflammatory lesions.',
         'wh_count', fv('whiteheads', 'count')),
        ('Papules', pa, '%',
         'Small solid raised lesions < 5 mm. Indicate active bacterial-driven inflammation.',
         'pa_count', fv('papules', 'count')),
        ('Pustules', pu, '%',
         'Pus-filled lesions with visible white/yellow apex. Active infection marker.',
         'pu_count', fv('pustules', 'count')),
        ('Erythema / Redness', rd, '/100',
         'Skin redness from vasodilation and inflammatory cytokine activity.',
         None, None),
        ('Post-Inflammatory Hyperpigmentation', hp, '%',
         'Residual dark marks after lesion resolution. Common on darker skin tones.',
         'hp_count', fv('hyperpigmentation', 'count')),
        ('Skin Texture Irregularity', tx, '/100',
         'Surface roughness analysis. Elevated scores indicate scarring or active inflammation.',
         None, None),
    ]

    feat_html = ''
    for name, val, unit, note, _, _ in feature_data:
        lbl, clr, bar_w = _level(val)
        feat_html += _feature_row(name, val, unit, lbl, clr, bar_w, note)

    # ── Lesion count summary ──────────────────────────────
    bh_count = int(fv('blackheads', 'count'))
    wh_count = int(fv('whiteheads', 'count'))
    pa_count = int(fv('papules',    'count'))
    pu_count = int(fv('pustules',   'count'))
    hp_count = int(fv('hyperpigmentation', 'count'))
    total_inflammatory     = pa_count + pu_count
    total_non_inflammatory = bh_count + wh_count
    total_lesions          = total_inflammatory + total_non_inflammatory

    lesion_count_html = f"""
    <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:24px;">
      {''.join([
        f'<div style="background:#f8f4ef; border:1px solid #ddd5c4; border-radius:10px; padding:14px 10px; text-align:center;">'
        f'<div style="font-size:1.6em; font-weight:800; color:#1e2d3d;">{count}</div>'
        f'<div style="font-size:0.72em; color:#8a8a8a; margin-top:3px; font-weight:600;">{label}</div>'
        f'</div>'
        for count, label in [
            (bh_count, 'Blackheads'),
            (wh_count, 'Whiteheads'),
            (pa_count, 'Papules'),
            (pu_count, 'Pustules'),
            (hp_count, 'PIH Spots'),
        ]
      ])}
    </div>
    <div style="display:flex; gap:16px; margin-bottom:24px; flex-wrap:wrap;">
      <div style="flex:1; min-width:160px; background:#fff7ed; border:1px solid #fdba74; border-radius:8px; padding:12px 16px;">
        <div style="font-size:0.72em; font-weight:700; color:#c2410c; text-transform:uppercase; letter-spacing:1px;">Total Inflammatory</div>
        <div style="font-size:1.8em; font-weight:800; color:#c2410c;">{total_inflammatory}</div>
        <div style="font-size:0.75em; color:#7a4a2a;">Papules + Pustules</div>
      </div>
      <div style="flex:1; min-width:160px; background:#eff6ff; border:1px solid #93c5fd; border-radius:8px; padding:12px 16px;">
        <div style="font-size:0.72em; font-weight:700; color:#1d4ed8; text-transform:uppercase; letter-spacing:1px;">Non-Inflammatory</div>
        <div style="font-size:1.8em; font-weight:800; color:#1d4ed8;">{total_non_inflammatory}</div>
        <div style="font-size:0.75em; color:#3a5a9a;">Blackheads + Whiteheads</div>
      </div>
      <div style="flex:1; min-width:160px; background:#f8f4ef; border:1px solid #ddd5c4; border-radius:8px; padding:12px 16px;">
        <div style="font-size:0.72em; font-weight:700; color:#1e2d3d; text-transform:uppercase; letter-spacing:1px;">Total Estimated Lesions</div>
        <div style="font-size:1.8em; font-weight:800; color:#1e2d3d;">{total_lesions}</div>
        <div style="font-size:0.75em; color:#5a5a5a;">Statistical estimate only</div>
      </div>
    </div>"""

    # ── Affected zones ────────────────────────────────────
    zones_html = """
    <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-bottom:6px;">"""

    zone_names  = ['Forehead', 'Cheeks', 'Nose', 'Chin', 'Jawline']
    forehead_pct = min(100, int((wh * 0.5) + (pa * 0.3) + (rd * 0.2)))
    cheeks_pct   = min(100, int((pa * 0.4) + (rd * 0.35) + (hp * 0.25)))
    nose_pct     = min(100, int((bh * 0.7) + (pu * 0.3)))
    chin_pct     = min(100, int((bh * 0.3) + (pa * 0.4) + (wh * 0.3)))
    jawline_pct  = min(100, int((rd * 0.5) + (pa * 0.3) + (hp * 0.2)))
    zone_pcts    = [forehead_pct, cheeks_pct, nose_pct, chin_pct, jawline_pct]

    for zone, pct in zip(zone_names, zone_pcts):
        if raw_class == 'clear_skin':
            pct = 0
        color  = '#dc2626' if pct >= 70 else '#ea580c' if pct >= 40 else '#d97706' if pct >= 15 else '#059669'
        status = 'High' if pct >= 70 else 'Mod.' if pct >= 40 else 'Low' if pct >= 15 else 'Clear'
        zones_html += f"""
      <div style="background:#f8f4ef; border:1px solid #ddd5c4; border-radius:10px; padding:12px 8px; text-align:center;">
        <div style="font-size:0.7em; font-weight:700; color:#8a8a8a; margin-bottom:6px;">{zone.upper()}</div>
        <div style="font-size:1.3em; font-weight:800; color:{color};">{pct}%</div>
        <div style="background:#ede7dd; border-radius:3px; height:5px; overflow:hidden; margin:6px 0;">
          <div style="width:{pct}%; height:100%; background:{color}; border-radius:3px;"></div>
        </div>
        <div style="font-size:0.7em; color:{color}; font-weight:600;">{status}</div>
      </div>"""

    zones_html += "</div>"

    # ── AI Probability bars ───────────────────────────────
    prob_colors = {'clear_skin':'#059669','mild':'#d97706','moderate':'#ea580c','severe':'#dc2626'}
    prob_labels = {'clear_skin':'Clear Skin','mild':'Mild Acne','moderate':'Moderate Acne','severe':'Severe Acne'}
    prob_html = ''
    for cls, pct in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        pct_val = round(pct * 100, 1) if pct <= 1 else round(pct, 1)
        clr     = prob_colors.get(cls, '#8a8a8a')
        lbl     = prob_labels.get(cls, cls)
        is_top  = cls == raw_class
        bg      = '#f8f4ef' if is_top else 'transparent'
        border  = f'border:1px solid {clr}40;' if is_top else ''
        prob_html += f"""
        <div style="display:flex; align-items:center; gap:14px; padding:10px 14px;
                    background:{bg}; border-radius:8px; {border} margin-bottom:6px;">
          <div style="width:140px; font-size:0.85em; font-weight:{'700' if is_top else '500'};
                      color:{'#1a1a1a' if is_top else '#5a5a5a'}; flex-shrink:0;">{lbl}</div>
          <div style="flex:1; background:#ede7dd; border-radius:4px; height:9px; overflow:hidden;">
            <div style="width:{pct_val}%; height:100%; background:{clr}; border-radius:4px;"></div>
          </div>
          <div style="width:48px; text-align:right; font-size:0.88em; font-weight:700; color:{clr};">{pct_val}%</div>
          {'<div style="font-size:0.72em; color:'+clr+'; font-weight:700; margin-left:6px;">&#10003; Primary</div>' if is_top else ''}
        </div>"""

    # ── Clinical Recommendations (change per classification) ──
    recs = {
        'clear_skin': [
            ('Maintenance Routine',
             'Continue current gentle skincare regimen. Use a mild non-comedogenic cleanser twice daily.',
             'Preventive care'),
            ('Sun Protection',
             'Apply broad-spectrum SPF 30+ every morning to prevent post-inflammatory pigmentation.',
             'Daily essential'),
            ('Barrier Preservation',
             'Use a non-comedogenic moisturiser morning and evening to maintain skin barrier integrity.',
             'Skin health'),
            ('Monitoring',
             'Re-analyse skin every 4–6 weeks. Seek professional assessment if new lesions appear.',
             'Follow-up'),
        ],
        'mild': [
            ('Topical Cleanser',
             'Salicylic acid (2%) face wash twice daily to reduce comedonal activity and unclog pores.',
             'First-line OTC'),
            ('Targeted Treatment',
             'Benzoyl peroxide (2.5–5%) as spot treatment on active lesions. Avoid periorbital area.',
             'First-line OTC'),
            ('Moisturisation',
             'Oil-free, non-comedogenic moisturiser after each cleanse to prevent dryness and barrier disruption.',
             'Daily essential'),
            ('Sun Protection',
             'SPF 30+ every morning — UV exposure worsens post-acne hyperpigmentation.',
             'Daily essential'),
            ('Treatment Timeline',
             'Allow 6–8 weeks of consistent application before evaluating efficacy. If no improvement, refer for prescription options.',
             'Clinical note'),
        ],
        'moderate': [
            ('Dermatologist Referral',
             'Prescribe topical retinoid (tretinoin 0.025–0.05%) or adapalene (0.1–0.3%) for comedonal control.',
             'Priority action'),
            ('Topical Antibiotic',
             'Consider clindamycin 1% or erythromycin combined with benzoyl peroxide to reduce resistance risk.',
             'Prescription'),
            ('Oral Antibiotic',
             'Doxycycline 50–100 mg daily or minocycline 50–100 mg daily for 3–6 months if widespread inflammatory lesions.',
             'Prescription'),
            ('PIH Management',
             'Azelaic acid 15–20% or niacinamide 4–10% to address existing post-inflammatory hyperpigmentation.',
             'Adjunct therapy'),
            ('Lifestyle Modifications',
             'Advise against picking lesions. Low-glycaemic diet may reduce sebum production. Stress management.',
             'Patient guidance'),
        ],
        'severe': [
            ('Isotretinoin Assessment',
             'Oral isotretinoin (0.5–1 mg/kg/day) is the most effective long-term treatment for severe acne. Requires pre-treatment blood panel and contraceptive counselling in females.',
             'Priority referral'),
            ('Intralesional Corticosteroid',
             'Triamcinolone acetonide 2.5–5 mg/mL injection for large, painful nodules/cysts to reduce inflammation rapidly.',
             'In-clinic procedure'),
            ('Combination Oral Therapy',
             'While awaiting isotretinoin: oral doxycycline + topical clindamycin/BPO as bridging therapy.',
             'Bridging treatment'),
            ('Scar Prevention',
             'Avoid all manipulation of lesions. Early intervention reduces risk of permanent atrophic or hypertrophic scarring.',
             'Critical guidance'),
            ('Psychological Support',
             'Severe acne significantly impacts quality of life and self-esteem. Screen for depression/anxiety. Refer as needed.',
             'Holistic care'),
        ],
    }

    rec_list = recs.get(raw_class, recs['mild'])
    rec_html = ''
    for i, (title, desc, tag) in enumerate(rec_list, 1):
        tag_color = '#c2410c' if tag in ('Priority action', 'Priority referral') else '#1d4ed8' if tag in ('Prescription', 'Bridging treatment', 'In-clinic procedure') else '#059669' if tag == 'Daily essential' else '#5a5a5a'
        rec_html += f"""
        <div style="display:flex; gap:14px; align-items:flex-start;
                    padding:14px 16px; background:#f8f4ef;
                    border:1px solid #ddd5c4; border-left:4px solid {sev['color']};
                    border-radius:0 10px 10px 0; margin-bottom:10px;">
          <div style="width:26px; height:26px; border-radius:50%;
                      background:{sev['color']}; color:white; font-size:0.75em;
                      font-weight:700; display:flex; align-items:center;
                      justify-content:center; flex-shrink:0; margin-top:1px;">{i}</div>
          <div style="flex:1;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px; flex-wrap:wrap;">
              <div style="font-weight:700; font-size:0.9em; color:#1e2d3d;">{title}</div>
              <span style="font-size:0.7em; padding:1px 8px; border-radius:8px;
                           background:{tag_color}18; color:{tag_color};
                           border:1px solid {tag_color}40; font-weight:600; white-space:nowrap;">{tag}</span>
            </div>
            <div style="font-size:0.85em; color:#5a5a5a; line-height:1.65;">{desc}</div>
          </div>
        </div>"""

    # ── Methodology note (changes per classification) ─────
    methodology_note = {
        'clear_skin': 'No significant lesion activity detected. Classification confidence was high with minimal overlap between severity classes.',
        'mild'      : 'Classification confidence indicates a clear mild presentation. Combined prediction threshold was not triggered.',
        'moderate'  : 'Moderate classification with possible borderline overlap with mild or severe. Clinical corroboration is recommended.',
        'severe'    : 'High-confidence severe classification. Multiple severity markers were elevated simultaneously across facial zones.',
    }.get(raw_class, '')

    # ── Face image ────────────────────────────────────────
    face_html = ''
    if face_img:
        face_html = f"""
        <div style="text-align:center;">
          <img src="/static/uploads/{face_img}"
               alt="Analysed Facial Region"
               style="max-width:200px; max-height:220px; object-fit:cover;
                      border-radius:12px; border:2px solid #ddd5c4;
                      box-shadow:0 4px 16px rgba(0,0,0,0.1);">
          <div style="font-size:0.75em; color:#8a8a8a; margin-top:8px;">
            Detected facial region
          </div>
        </div>"""

    # ── Health score ring (SVG) ───────────────────────────
    score_color = '#059669' if hs >= 70 else '#d97706' if hs >= 40 else '#dc2626'

    # ══════════════════════════════════════════════════════
    # FULL HTML REPORT
    # ══════════════════════════════════════════════════════
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>AcneVision Medical Report — {report_id}</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=DM+Serif+Display&display=swap" rel="stylesheet">
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family:'DM Sans',sans-serif; background:#f0ebe3; color:#1a1a1a; }}
    @media print {{
      body {{ background:white; }}
      .no-print {{ display:none !important; }}
      .page {{ box-shadow:none !important; margin:0 !important; border-radius:0 !important; }}
      @page {{ margin:1.5cm; }}
    }}
    table {{ border-collapse:collapse; width:100%; }}
  </style>
</head>
<body>

<!-- TOOLBAR -->
<div class="no-print" style="position:fixed; top:0; left:0; right:0; z-index:100;
  background:#1e2d3d; padding:10px 24px; display:flex; align-items:center;
  justify-content:space-between; box-shadow:0 2px 12px rgba(0,0,0,0.3);">
  <div style="color:white; font-weight:700; font-size:0.92em;">
    AcneVision &nbsp;<span style="opacity:0.5; font-weight:400;">{report_id}</span>
  </div>
  <div style="display:flex; gap:8px;">
    <button onclick="window.print()" style="padding:8px 20px; background:#c1674a; color:white;
      border:none; border-radius:7px; font-size:0.85em; font-weight:600;
      cursor:pointer; font-family:'DM Sans',sans-serif;">Print / Save PDF</button>
    <button onclick="window.close()" style="padding:8px 16px; background:rgba(255,255,255,0.1); color:white;
      border:1px solid rgba(255,255,255,0.2); border-radius:7px; font-size:0.85em;
      cursor:pointer; font-family:'DM Sans',sans-serif;">Close</button>
  </div>
</div>

<div style="height:52px;" class="no-print"></div>

<!-- PAGE -->
<div class="page" style="max-width:900px; margin:28px auto 48px; background:white;
  border-radius:16px; overflow:hidden;
  box-shadow:0 8px 40px rgba(0,0,0,0.12);">

  <!-- TOP HEADER BAND -->
  <div style="background:#1e2d3d; padding:32px 40px;">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
      <div>
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
          <div style="width:42px; height:42px; background:#c1674a; border-radius:10px;
                      display:flex; align-items:center; justify-content:center;
                      color:white; font-weight:800; font-size:1em;">AV</div>
          <div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.4em; color:white;">AcneVision</div>
            <div style="font-size:0.7em; color:rgba(255,255,255,0.5); letter-spacing:1.5px; text-transform:uppercase;">Dermatological Assessment Platform</div>
          </div>
        </div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.9em; color:white; line-height:1.2;">
          Skin Analysis Report
        </div>
        <div style="font-size:0.82em; color:rgba(255,255,255,0.55); margin-top:6px;">
          For review by a qualified dermatologist or skin health professional
        </div>
      </div>
      <div style="text-align:right;">
        <div style="background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15);
                    border-radius:10px; padding:14px 18px; font-size:0.82em; line-height:2; color:rgba(255,255,255,0.75);">
          <div><span style="color:rgba(255,255,255,0.45);">Report ID</span> &nbsp; <strong style="color:white;">{report_id}</strong></div>
          <div><span style="color:rgba(255,255,255,0.45);">Date</span> &nbsp; <strong style="color:white;">{date_str}</strong></div>
          <div><span style="color:rgba(255,255,255,0.45);">Method</span> &nbsp; <strong style="color:white;">Image Analysis System</strong></div>
          <div><span style="color:rgba(255,255,255,0.45);">ICD-10</span> &nbsp; <strong style="color:white;">{sev['icd_code']}</strong></div>
        </div>
      </div>
    </div>
  </div>

  <!-- CLINICAL REFERENCE NOTICE -->
  <div style="background:#fef9c3; border-bottom:1px solid #fcd34d; padding:10px 40px;
              display:flex; align-items:center; gap:10px;">
    <span style="font-size:1.1em;">&#9888;&#65039;</span>
    <div style="font-size:0.8em; color:#854d0e; line-height:1.5;">
      <strong>Clinical reference notice.</strong> This report provides an image-based assessment of visible skin characteristics.
      Results are intended for informational and screening purposes only and should be interpreted in conjunction with
      professional clinical evaluation. Assessment outcomes may be affected by image quality, lighting conditions, and facial positioning.
    </div>
  </div>

  <div style="padding:36px 40px;">

    <!-- SECTION 1: PRIMARY DIAGNOSIS -->
    {_section_title('1. Primary Diagnosis', 'Acne severity assessment')}
    <div style="background:{sev['bg']}; border:2px solid {sev['border']};
                border-left:6px solid {sev['color']}; border-radius:12px;
                padding:24px 28px; margin-bottom:28px;">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
        <div style="flex:1; min-width:260px;">
          <div style="font-size:0.72em; font-weight:700; color:{sev['color']};
                      text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;">
            {sev['grade']}
          </div>
          <div style="font-family:'DM Serif Display',serif; font-size:2.4em;
                      color:{sev['color']}; line-height:1.1; margin-bottom:12px;">
            {severity}
          </div>
          <div style="font-size:0.87em; color:#444; line-height:1.7; max-width:440px;">
            {sev['clinical']}
          </div>
          <div style="margin-top:16px; display:flex; gap:12px; flex-wrap:wrap;">
            <div style="padding:6px 14px; background:{sev['urgency_bg']};
                        color:{sev['urgency_color']}; border-radius:8px;
                        font-size:0.8em; font-weight:700; border:1px solid {sev['color']}40;">
              {sev['urgency']}
            </div>
            <div style="padding:6px 14px; background:#f8f4ef;
                        color:#5a5a5a; border-radius:8px;
                        font-size:0.8em; border:1px solid #ddd5c4;">
              GEA Scale: {sev['gea_scale']}
            </div>
          </div>
        </div>
        <div style="text-align:center; min-width:140px;">
          <svg width="120" height="120" viewBox="0 0 120 120" style="transform:rotate(-90deg); display:block; margin:0 auto;">
            <circle cx="60" cy="60" r="50" fill="none" stroke="#e8e0d5" stroke-width="12"/>
            <circle cx="60" cy="60" r="50" fill="none" stroke="{score_color}" stroke-width="12"
                    stroke-linecap="round" stroke-dasharray="314"
                    stroke-dashoffset="{314 - (314 * int(hs) / 100):.0f}"/>
          </svg>
          <div style="margin-top:-66px; font-size:1.9em; font-weight:800; color:#1e2d3d;">{int(hs)}</div>
          <div style="font-size:0.7em; color:#8a8a8a; margin-top:44px;">Skin Health Score</div>
          <div style="font-size:0.68em; color:#8a8a8a;">/ 100</div>
          <div style="margin-top:10px; font-size:0.72em; font-weight:700; color:{sev['color']};
                      background:{sev['urgency_bg']}; padding:4px 10px; border-radius:6px;">
            Assessment confidence: {conf_pct}%
          </div>
        </div>
      </div>
      <div style="margin-top:16px; padding-top:14px; border-top:1px solid {sev['border']};
                  font-size:0.82em; color:#5a5a5a; line-height:1.6;">
        <strong style="color:#1a1a1a;">Prognosis:</strong> {sev['prognosis']}
      </div>
    </div>

    <!-- SECTION 2: PATIENT IMAGE -->
    {"" if not face_img else f'''
    <div style="margin-bottom:28px;">
      {_section_title("2. Analysed Image", "Facial image prepared for dermatological assessment")}
      <div style="background:#f8f4ef; border:1px solid #ddd5c4; border-radius:12px;
                  padding:20px; display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
        {face_html}
        <div style="flex:1; min-width:200px; font-size:0.85em; color:#5a5a5a; line-height:1.8;">
          <div style="margin-bottom:8px;"><strong style="color:#1a1a1a;">Image preprocessing:</strong><br>
            Face detected and cropped for image analysis input.</div>
          <div style="margin-bottom:8px;"><strong style="color:#1a1a1a;">Heatmap:</strong><br>
            Gradient activation map generated to highlight regions most influential in the severity prediction.</div>
          <div><strong style="color:#1a1a1a;">Note:</strong><br>
            Image quality, lighting, and skin tone may affect accuracy. Best results with even lighting on a clean face.</div>
        </div>
      </div>
    </div>
    '''}

    <!-- SECTION 3: CLASSIFICATION PROBABILITIES -->
    {_section_title("3. Classification Probabilities", "Confidence distribution across all four severity classes")}
    <div style="background:#f8f4ef; border:1px solid #ddd5c4; border-radius:12px;
                padding:20px 24px; margin-bottom:28px;">
      {prob_html}
      <div style="font-size:0.75em; color:#8a8a8a; margin-top:10px; font-style:italic; padding:0 4px;">
        A combined prediction is returned when the top two class probabilities differ by less than 30%,
        indicating borderline presentation between adjacent severity grades.
        The primary diagnosis above uses the highest-confidence class.
      </div>
    </div>

    <!-- SECTION 4: LESION COUNT SUMMARY -->
    {_section_title("4. Estimated Lesion Count", "Statistical estimates — not equivalent to manual clinical counting")}
    {lesion_count_html}

    <!-- SECTION 5: FACIAL ZONE DISTRIBUTION -->
    {_section_title("5. Facial Zone Distribution", "Estimated acne activity percentage per region")}
    <div style="margin-bottom:28px;">
      {zones_html}
      <div style="font-size:0.75em; color:#8a8a8a; margin-top:8px; font-style:italic;">
        Zone percentages are derived from feature densities and are indicative estimates, not clinical zone mapping.
        Clinician should verify affected areas during physical examination.
      </div>
    </div>

    <!-- SECTION 6: SKIN FEATURE ANALYSIS -->
    {_section_title("6. Detailed Skin Feature Analysis", "Seven markers measured via image analysis")}
    <div style="border:1px solid #ddd5c4; border-radius:12px; overflow:hidden; margin-bottom:28px;">
      <table>
        <thead>
          <tr style="background:#1e2d3d;">
            <th style="padding:11px 16px; text-align:left; font-size:0.74em; font-weight:700;
                       color:rgba(255,255,255,0.65); text-transform:uppercase; letter-spacing:1px;">Feature</th>
            <th style="padding:11px 16px; text-align:center; font-size:0.74em; font-weight:700;
                       color:rgba(255,255,255,0.65); text-transform:uppercase; letter-spacing:1px;">Score</th>
            <th style="padding:11px 16px; text-align:left; font-size:0.74em; font-weight:700;
                       color:rgba(255,255,255,0.65); text-transform:uppercase; letter-spacing:1px;">Level</th>
            <th style="padding:11px 16px; text-align:center; font-size:0.74em; font-weight:700;
                       color:rgba(255,255,255,0.65); text-transform:uppercase; letter-spacing:1px;">Grade</th>
            <th style="padding:11px 16px; text-align:left; font-size:0.74em; font-weight:700;
                       color:rgba(255,255,255,0.65); text-transform:uppercase; letter-spacing:1px;">Clinical note</th>
          </tr>
        </thead>
        <tbody>
          {feat_html}
        </tbody>
      </table>
    </div>

    <!-- SECTION 7: CLINICAL RECOMMENDATIONS -->
    {_section_title("7. Clinical Recommendations", f"Evidence-based guidance for {severity.lower()}")}
    <div style="margin-bottom:28px;">
      {rec_html}
    </div>

    <!-- SECTION 8: METHODOLOGY -->
    {_section_title("8. Analysis Methodology", "Technical details for clinical interpretation")}
    <div style="background:#f8f4ef; border:1px solid #ddd5c4; border-radius:12px;
                padding:22px 24px; margin-bottom:28px;">
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;
                  font-size:0.83em; color:#5a5a5a; line-height:1.75;">
        <div>
          <strong style="color:#1a1a1a; display:block; margin-bottom:4px;">Classification Model</strong>
          A multi-class severity classification model trained on a balanced
          dermatological image dataset (clear / mild / moderate / severe).
          Input resolution: 224 x 224 px.
        </div>
        <div>
          <strong style="color:#1a1a1a; display:block; margin-bottom:4px;">Feature Detection Pipeline</strong>
          Rule-based image analysis: colour-space segmentation for redness
          and hyperpigmentation; contour detection for lesion counting;
          texture matrix analysis for surface roughness scoring.
        </div>
        <div>
          <strong style="color:#1a1a1a; display:block; margin-bottom:4px;">Training Data</strong>
          Balanced dermatological image dataset with augmentation applied
          (flips, brightness, rotation). Performance evaluated on a
          held-out validation set.
        </div>
        <div>
          <strong style="color:#1a1a1a; display:block; margin-bottom:4px;">Activation Mapping</strong>
          Gradient-weighted activation mapping applied to produce a heatmap
          visualising which facial regions most influenced the
          severity prediction.
        </div>
        <div>
          <strong style="color:#1a1a1a; display:block; margin-bottom:4px;">Lesion Count Estimation</strong>
          Region detection on pre-processed images. Counts are statistical
          estimates subject to lighting, skin tone, and image resolution
          variability.
        </div>
        <div>
          <strong style="color:#1a1a1a; display:block; margin-bottom:4px;">Severity Classification Scale</strong>
          Aligned to the Global Evaluation of Acne (GEA) scale and IGA
          (Investigator's Global Assessment). Combined prediction issued
          when class probability gap &lt; 30%.
        </div>
      </div>
      {f'<div style="margin-top:16px; padding-top:14px; border-top:1px solid #ddd5c4; font-size:0.82em; color:#5a5a5a; font-style:italic;">{methodology_note}</div>' if methodology_note else ''}
    </div>

    <!-- SECTION 9: MEDICAL DISCLAIMER -->
    <div style="background:#fef3c7; border:1px solid #fcd34d; border-left:5px solid #f59e0b;
                border-radius:0 10px 10px 0; padding:18px 22px; margin-bottom:32px;">
      <div style="font-weight:700; color:#92400e; font-size:0.9em; margin-bottom:8px;">
        &#9888;&#65039; Medical Disclaimer &amp; Limitations
      </div>
      <div style="font-size:0.82em; color:#78350f; line-height:1.75;">
        This report is generated for <strong>clinical reference and screening purposes only</strong>
        and does not constitute a medical diagnosis, clinical prescription, or professional medical advice.
        Image-based skin analysis is inherently limited by image quality, ambient lighting, camera sensor,
        and patient skin tone.
        <br><br>
        Lesion counts, zone percentages, and feature scores are <strong>statistical estimates</strong>
        derived from a single photograph and cannot replace clinical counting or dermoscopy.
        The severity classification should be corroborated by physical examination.
        <br><br>
        <strong>Always consult a qualified, board-certified dermatologist</strong> for proper diagnosis
        and individualised treatment planning. Do not delay or substitute professional medical care
        based on this report.
      </div>
    </div>

    <!-- FOOTER -->
    <div style="border-top:1px solid #ddd5c4; padding-top:20px;
                display:flex; justify-content:space-between; align-items:center;
                flex-wrap:wrap; gap:12px;">
      <div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.1em; color:#1e2d3d;">AcneVision</div>
        <div style="font-size:0.73em; color:#8a8a8a; margin-top:2px;">
          Dermatological Assessment Platform &middot; 2026
        </div>
      </div>
      <div style="text-align:right; font-size:0.75em; color:#8a8a8a; line-height:1.8;">
        <div>Report ID: <strong style="color:#1a1a1a;">{report_id}</strong></div>
        <div>Generated: {timestamp}</div>
        <div style="margin-top:4px; padding:3px 10px; background:#f8f4ef;
                    border:1px solid #ddd5c4; border-radius:5px;
                    color:#5a5a5a; font-weight:600; font-size:0.9em; display:inline-block;">
          Clinical Reference Only
        </div>
      </div>
    </div>

  </div><!-- /padding div -->
</div><!-- /page -->

</body>
</html>"""

    return html