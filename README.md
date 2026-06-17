# AcneVision 🔬
### AI-Powered Acne Severity Detection & Skin Analysis System

> A Final Year Project — Bachelor of Science in Artificial Intelligence and Machine Learning  
> International Business, Science and Technology University (ISBAT), Kampala, Uganda — 2026

---

## What is AcneVision?

AcneVision is an intelligent web-based system that uses deep learning to automatically detect and classify acne severity from facial photographs. Users upload a photo or use their webcam and receive an instant clinical-grade skin analysis report, personalised skincare recommendations, and access to an AI skincare assistant.

---

## Features

| Feature | Description |
|---|---|
| 🔍 Acne Severity Detection | CNN classifies skin into Clear, Mild, Moderate, or Severe |
| 🗺️ Grad-CAM Heatmap | Visual explanation showing which areas influenced the prediction |
| 📊 Skin Feature Analysis | 7 features analysed: blackheads, whiteheads, papules, pustules, redness, hyperpigmentation, texture |
| 📋 Medical Report | Downloadable clinical-grade HTML report |
| 🤖 AI Skincare Assistant | Ollama-powered chatbot (llama3.2:3b) for personalised advice |
| ✨ Gemini AI Summaries | Google Gemini generates professional text explanations |
| 📍 Clinic Locator | Find 12 dermatology clinics in Kampala with map and directions |
| 📚 Learn Page | Interactive skin education with modals and quizzes |
| 🕓 Analysis History | All past scans saved and reviewable |
| 🌙 Dark / Light Theme | Toggle between themes on every page |
| 📸 Webcam Support | Capture and analyse directly from your camera |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10, Flask, Flask-Login, Flask-SQLAlchemy |
| AI / ML | TensorFlow 2.x, Keras, MobileNetV2 (Transfer Learning) |
| Computer Vision | OpenCV, Grad-CAM, HSV/LAB colour analysis |
| Database | SQLite (via SQLAlchemy ORM) |
| AI Assistant | Ollama + llama3.2:3b (local, offline) |
| AI Text | Google Gemini 2.0 Flash API |
| Frontend | HTML5, CSS3, Vanilla JavaScript, DM Sans + Playfair Display fonts |

---

## Project Structure

```
AcneVision/
│
├── app.py                          # Main Flask application — all routes
├── requirements.txt                # Python dependencies
├── .env                            # API keys (not committed to git)
├── README.md                       # This file
│
├── chatbot/
│   ├── __init__.py
│   └── simple_bot.py               # Ollama llama3.2:3b integration
│
├── utils/
│   ├── __init__.py
│   ├── face_detection.py           # OpenCV Haar Cascade face detection
│   ├── prediction.py               # CNN inference + combined prediction logic
│   ├── gradcam.py                  # Grad-CAM heatmap generator
│   ├── skin_features.py            # 7-feature skin analysis
│   ├── lesion_detector.py          # OpenCV lesion annotation
│   ├── skin_explainer.py           # Evidence explanation engine
│   ├── report_generator.py         # Medical HTML report generator
│   ├── routine_generator.py        # Skincare routine generator
│   └── gemini_text_generator.py    # Google Gemini AI text generation
│
├── training/
│   └── train.py                    # MobileNetV2 training script (2-phase)
│
├── model/
│   └── acne_model.keras            # Trained CNN model
│
├── dataset/
│   ├── train/
│   │   ├── clear_skin/             # ~210 training images
│   │   ├── mild/                   # ~210 training images
│   │   ├── moderate/               # ~210 training images
│   │   └── severe/                 # ~210 training images
│   └── validation/
│       ├── clear_skin/
│       ├── mild/
│       ├── moderate/
│       └── severe/
│
├── database/
│   └── app.db                      # SQLite database (auto-created)
│
├── static/
│   ├── css/
│   │   └── theme.css               # Global dark/light theme system
│   ├── js/
│   │   └── theme.js                # Theme toggle logic
│   ├── uploads/                    # User uploaded and cropped face images
│   ├── heatmaps/                   # Grad-CAM output images
│   └── reports/                    # Generated HTML medical reports
│
└── templates/
    ├── base.html                   # Shared nav, footer, theme button
    ├── index.html                  # Hero upload page (Skinity-inspired)
    ├── results.html                # Analysis results dashboard
    ├── login.html                  # Split-screen login (Skinity-style)
    ├── register.html               # Split-screen register
    ├── history.html                # Analysis history page
    ├── learn.html                  # Interactive skin education
    └── clinics.html                # Kampala clinic locator with map
```

---

## Setup & Installation

### Requirements
- Python 3.10+
- Windows 10/11 (tested) or Linux/macOS
- At least 8 GB RAM
- At least 10 GB free disk space
- Ollama installed (for AI chatbot)

---

### Step 1 — Clone or Download the Project

Place the project folder at:
```
C:\Users\pc\OneDrive\Attachments\Desktop\AcneVision\
```

---

### Step 2 — Create a Virtual Environment

```bash
cd AcneVision
python -m venv venv
venv\Scripts\activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Set Up Environment Variables

Create a file called `.env` in the root folder:

```dotenv
GEMINI_API_KEY=AIzaSy...your_key_here
GEMINI_MODEL=gemini-2.0-flash
```

Get your free Gemini API key at: https://aistudio.google.com/apikey

> The app works without Gemini — it falls back to built-in summaries automatically.

---

### Step 5 — Install and Start Ollama (AI Chatbot)

Download Ollama from: https://ollama.com/download

Then run in a terminal:
```bash
ollama serve
ollama pull llama3.2:3b
```

> If you see "address already in use" — Ollama is already running, ignore it.

---

### Step 6 — Add Your Trained Model

Place your trained model file at:
```
AcneVision/model/acne_model.keras
```

If you need to train from scratch:
```bash
cd training
python train.py
```

---

### Step 7 — Run the Application

```bash
# Make sure venv is active
venv\Scripts\activate

# Start the app
python app.py
```

Open your browser and go to:
```
http://127.0.0.1:5000
```

---

## How to Use

1. **Register** a new account at `/register`
2. **Log in** at `/login`
3. **Upload a photo** or use your **webcam** on the home page
4. Wait a few seconds for the AI to analyse your skin
5. View your **results** — severity, heatmap, features, recommendations
6. Download your **medical report** as HTML
7. Chat with the **AI assistant** for personalised advice
8. Find a **clinic** near you in Kampala
9. Visit the **Learn** page to understand your skin better
10. Check your **History** to track changes over time

---

## Tips for Best Results

- Use a **clear, well-lit frontal face photo**
- Natural or indoor lighting works best
- **No heavy makeup**, filters, or face coverings
- Face should be **centred and fully visible**
- Selfie distance (arm's length) gives most accurate results
- Supported formats: **JPG, PNG, WEBP, BMP** (max 16 MB)

---

## Model Architecture

| Detail | Value |
|---|---|
| Base Model | MobileNetV2 (ImageNet pre-trained) |
| Input Size | 224 × 224 × 3 RGB |
| Classes | 4 (Clear Skin, Mild, Moderate, Severe) |
| Training Phases | 2 (frozen base → fine-tune last 50 layers) |
| Augmentation | Rotation, zoom, flip, brightness, channel shift |
| Class Weighting | Balanced (computed from training distribution) |
| Output | Softmax probabilities per class |
| Explainability | Grad-CAM heatmap + input gradient saliency |

---

## Known Limitations

- Accuracy depends heavily on **image quality and lighting**
- Model may struggle with **unusual angles** or facial obstructions
- **Dark skin tones** may show reduced accuracy (training dataset bias)
- AI chatbot requires **Ollama running locally** — slower on CPU
- This is a **research/educational tool** — not a medical diagnosis system
- Always consult a qualified dermatologist for professional medical advice

---

## Disclaimer

> AcneVision is developed for **educational and research purposes only**.  
> It does not constitute a medical diagnosis, prescription, or professional medical advice.  
> Always consult a board-certified dermatologist for proper skin assessment and treatment.

---

## Author

**Asiimwe Tracy Agnes**  
Roll Number: 012230037  
Bachelor of Science in Artificial Intelligence and Machine Learning  
International Business, Science and Technology University (ISBAT)  
Kampala, Uganda — 2026  
Supervisor: Mr. Umesh Kumar

---

*AcneVision — Know Your Skin. Transform Your Routine.*