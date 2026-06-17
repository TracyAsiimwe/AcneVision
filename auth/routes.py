<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AcneVision — {% if mode == 'register' %}Create Account{% else %}Sign In{% endif %}</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0a0a0f;
      --bg-2: #111118;
      --bg-3: #1a1a24;
      --card: #13131a;
      --border: rgba(255,255,255,0.08);
      --border-2: rgba(255,255,255,0.12);
      --text: #e8e8ec;
      --text-2: #9a9aa8;
      --text-3: #6a6a78;
      --rose: #c87a6b;
      --rose-d: #a85e50;
      --rose-glow: rgba(200,122,107,0.15);
      --gold: #c9a96e;
      --gold-d: #a88a50;
      --white: #ffffff;
    }
    * { margin:0; padding:0; box-sizing:border-box; }
    html { scroll-behavior:smooth; }
    body {
      font-family:'DM Sans',sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height:100vh;
      display:flex;
    }

    /* LEFT PANEL — Visual */
    .left-panel {
      width: 50%;
      background: linear-gradient(135deg, #0a0a0f 0%, #1a1218 50%, #0f0a0f 100%);
      display:flex; flex-direction:column;
      justify-content:center; align-items:center;
      padding: 60px;
      position:relative; overflow:hidden;
    }
    .left-panel::before {
      content:''; position:absolute; inset:0;
      background:
        radial-gradient(ellipse at 20% 50%, rgba(200,122,107,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 30%, rgba(201,169,110,0.05) 0%, transparent 50%);
      pointer-events:none;
    }
    .brand-mark {
      width:56px; height:56px;
      background: linear-gradient(135deg, var(--rose), var(--gold));
      border-radius:16px;
      display:flex; align-items:center; justify-content:center;
      color:white; font-weight:700; font-size:1.1em;
      margin-bottom:28px; position:relative; z-index:1;
      box-shadow: 0 8px 32px rgba(200,122,107,0.3);
    }
    .brand-name {
      font-family:'DM Serif Display',serif;
      font-size:2.2em; color:var(--white);
      margin-bottom:12px; position:relative; z-index:1;
      letter-spacing:-1px;
    }
    .brand-tag {
      font-size:0.9em; color:var(--text-2);
      max-width:320px; text-align:center; line-height:1.7;
      position:relative; z-index:1;
    }
    .brand-stats {
      display:flex; gap:40px; margin-top:48px;
      position:relative; z-index:1;
    }
    .stat {
      text-align:center;
    }
    .stat-num {
      font-family:'DM Serif Display',serif;
      font-size:1.8em; color:var(--gold);
      line-height:1;
    }
    .stat-label {
      font-size:0.72em; color:var(--text-3);
      margin-top:6px; text-transform:uppercase;
      letter-spacing:1.5px;
    }

    /* RIGHT PANEL — Form */
    .right-panel {
      width: 50%;
      display:flex; flex-direction:column;
      justify-content:center; align-items:center;
      padding: 60px;
      background: var(--bg);
    }
    .form-wrap {
      width:100%; max-width:400px;
    }
    .form-header {
      margin-bottom:32px;
    }
    .form-eyebrow {
      font-size:0.72em; font-weight:700;
      color:var(--rose); text-transform:uppercase;
      letter-spacing:2.5px; margin-bottom:10px;
    }
    .form-title {
      font-family:'DM Serif Display',serif;
      font-size:1.9em; color:var(--white);
      margin-bottom:8px; letter-spacing:-0.5px;
    }
    .form-sub {
      font-size:0.88em; color:var(--text-2);
      line-height:1.6;
    }

    /* Toggle tabs */
    .auth-tabs {
      display:flex; gap:0; margin-bottom:28px;
      background: var(--bg-3);
      border-radius:10px; padding:4px;
      border:1px solid var(--border);
    }
    .auth-tab {
      flex:1; padding:10px 0;
      text-align:center; font-size:0.85em;
      font-weight:600; color:var(--text-2);
      text-decoration:none; border-radius:8px;
      transition:all 0.25s; cursor:pointer;
      border:none; background:transparent;
      font-family:'DM Sans',sans-serif;
    }
    .auth-tab.active {
      background: var(--rose);
      color: white;
      box-shadow: 0 2px 12px rgba(200,122,107,0.3);
    }
    .auth-tab:hover:not(.active) {
      color: var(--text);
    }

    /* Fields */
    .field { margin-bottom:18px; }
    .field label {
      display:block; font-size:0.78em;
      font-weight:600; color:var(--text-2);
      margin-bottom:8px; text-transform:uppercase;
      letter-spacing:1px;
    }
    .field input {
      width:100%; padding:13px 16px;
      border:1.5px solid var(--border);
      border-radius:10px; font-size:0.92em;
      font-family:'DM Sans',sans-serif;
      background: var(--bg-2);
      color:var(--text); transition:all 0.25s;
      outline:none;
    }
    .field input::placeholder { color:var(--text-3); }
    .field input:focus {
      border-color: var(--rose);
      background: var(--bg-3);
      box-shadow: 0 0 0 3px var(--rose-glow);
    }

    /* Button */
    .btn {
      width:100%; padding:14px;
      background: linear-gradient(135deg, var(--rose), var(--rose-d));
      color:white; border:none; border-radius:10px;
      font-size:0.95em; font-weight:700;
      cursor:pointer; transition:all 0.25s;
      font-family:'DM Sans',sans-serif;
      margin-top:6px;
      box-shadow: 0 4px 20px rgba(200,122,107,0.25);
    }
    .btn:hover {
      transform:translateY(-2px);
      box-shadow: 0 8px 28px rgba(200,122,107,0.35);
    }
    .btn:active { transform:translateY(0); }

    /* Alerts */
    .alert {
      padding:12px 16px; border-radius:10px;
      margin-bottom:18px; font-size:0.85em;
      border:1px solid;
    }
    .alert-error {
      background:rgba(220,60,60,0.08);
      border-color:rgba(220,60,60,0.2);
      color:#e57373;
    }
    .alert-success {
      background:rgba(60,180,100,0.08);
      border-color:rgba(60,180,100,0.2);
      color:#81c784;
    }

    /* Divider */
    .divider {
      display:flex; align-items:center;
      gap:14px; margin:24px 0;
      color:var(--text-3); font-size:0.78em;
    }
    .divider::before, .divider::after {
      content:''; flex:1;
      border-top:1px solid var(--border);
    }

    /* Footer link */
    .form-footer {
      text-align:center; margin-top:24px;
      font-size:0.85em; color:var(--text-2);
    }
    .form-footer a {
      color:var(--rose); font-weight:600;
      text-decoration:none; transition:color 0.2s;
    }
    .form-footer a:hover { color:var(--gold); }

    /* Responsive */
    @media(max-width:900px){
      body { flex-direction:column; }
      .left-panel { width:100%; padding:40px 24px; min-height:280px; }
      .right-panel { width:100%; padding:40px 24px; }
      .brand-stats { gap:28px; }
    }
  </style>
<base target="_blank">
</head>
<body>

  <!-- LEFT PANEL -->
  <div class="left-panel">
    <div class="brand-mark">AV</div>
    <div class="brand-name">AcneVision</div>
    <div class="brand-tag">
      AI-powered skin analysis with deep learning precision.
      Track, understand, and improve your skin health.
    </div>
    <div class="brand-stats">
      <div class="stat">
        <div class="stat-num">4</div>
        <div class="stat-label">Severity Levels</div>
      </div>
      <div class="stat">
        <div class="stat-num">7</div>
        <div class="stat-label">Skin Features</div>
      </div>
      <div class="stat">
        <div class="stat-num">99%</div>
        <div class="stat-label">Accuracy</div>
      </div>
    </div>
  </div>

  <!-- RIGHT PANEL -->
  <div class="right-panel">
    <div class="form-wrap">

      <div class="form-header">
        <div class="form-eyebrow">{% if mode == 'register' %}Get Started{% else %}Welcome Back{% endif %}</div>
        <div class="form-title">{% if mode == 'register' %}Create Account{% else %}Sign In{% endif %}</div>
        <div class="form-sub">
          {% if mode == 'register' %}
            Start your personalized skin health journey
          {% else %}
            Access your analyses and AI skin insights
          {% endif %}
        </div>
      </div>

      <!-- Toggle Tabs -->
      <div class="auth-tabs">
        <a href="/auth?mode=login" class="auth-tab {% if mode != 'register' %}active{% endif %}">Sign In</a>
        <a href="/auth?mode=register" class="auth-tab {% if mode == 'register' %}active{% endif %}">Register</a>
      </div>

      {% with messages = get_flashed_messages(with_categories=true) %}
        {% for category, message in messages %}
        <div class="alert alert-{{ category }}">{{ message }}</div>
        {% endfor %}
      {% endwith %}

      {% if mode == 'register' %}
      <!-- REGISTER FORM -->
      <form method="POST" action="/auth?mode=register">
        <input type="hidden" name="action" value="register">
        <div class="field">
          <label>Username</label>
          <input type="text" name="username" placeholder="Choose a username" required>
        </div>
        <div class="field">
          <label>Email Address</label>
          <input type="email" name="email" placeholder="you@example.com" required>
        </div>
        <div class="field">
          <label>Password</label>
          <input type="password" name="password" placeholder="Min 6 characters" required>
        </div>
        <div class="field">
          <label>Confirm Password</label>
          <input type="password" name="confirm" placeholder="Repeat password" required>
        </div>
        <button type="submit" class="btn">Create Account</button>
      </form>
      <div class="form-footer">
        Already have an account? <a href="/auth?mode=login">Sign in</a>
      </div>

      {% else %}
      <!-- LOGIN FORM -->
      <form method="POST" action="/auth?mode=login">
        <input type="hidden" name="action" value="login">
        <div class="field">
          <label>Username</label>
          <input type="text" name="username" placeholder="Enter your username" required>
        </div>
        <div class="field">
          <label>Password</label>
          <input type="password" name="password" placeholder="Enter your password" required>
        </div>
        <button type="submit" class="btn">Sign In</button>
      </form>
      <div class="form-footer">
        Don't have an account? <a href="/auth?mode=register">Create one</a>
      </div>
      {% endif %}

    </div>
  </div>

</body>
</html>