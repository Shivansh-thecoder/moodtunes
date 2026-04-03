# ── 🔥 FIX OpenBLAS ──────────────────────────────────────────────────────────
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ── Imports ──────────────────────────────────────────────────────────────────
import joblib
import requests
import urllib.parse
import secrets
import numpy as np
from flask import Flask, request, jsonify, session, redirect, render_template
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")

SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI")

SPOTIFY_AUTH_URL  = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"

# ── Load ML model ─────────────────────────────────────────────────────────────
model, st_model, le = None, None, None

if os.path.exists("best_model.pkl") and os.path.exists("label_encoder.pkl"):
    model = joblib.load("best_model.pkl")
    le = joblib.load("label_encoder.pkl")
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print("✅ Model loaded!")
else:
    print("⚠️ Model files missing")

# ── Rule-based override ───────────────────────────────────────────────────────
def rule_based_override(text):
    text = text.lower()

    scores = {
        "joy": 0,
        "sadness": 0,
        "anger": 0,
        "fear": 0,
        "surprise": 0,
        "neutral": 0,
    }

    rules = {
        "joy": ["happy","excited","motivated","productive","confident","energetic","pumped","optimistic"],
        "sadness": ["sad","depressed","lonely","heartbroken","down","miserable","crying"],
        "anger": ["angry","mad","furious","annoyed","frustrated","hate","pissed"],
        "fear": ["anxious","stressed","worried","panic","nervous","overthinking","tense"],
        "surprise": ["confused","shocked","unexpected","wtf","curious"],
        "neutral": ["okay","fine","meh","bored","normal","idk","chilling"]
    }

    for mood, words in rules.items():
        for w in words:
            if w in text:
                scores[mood] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

# ── Mood → Spotify search queries ─────────────────────────────────────────────
MOOD_PARAMS = {
    "sadness": {"query": "sad acoustic songs", "label": "Melancholic", "emoji": "🌧"},
    "joy": {"query": "happy upbeat songs", "label": "Joyful", "emoji": "✨"},
    "anger": {"query": "intense rock songs", "label": "Intense", "emoji": "🔥"},
    "fear": {"query": "calm ambient music", "label": "Uneasy", "emoji": "🌙"},
    "surprise": {"query": "experimental music", "label": "Unexpected", "emoji": "😮"},
    "neutral": {"query": "chill pop music", "label": "Neutral", "emoji": "🎵"},
}

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", logged_in=("access_token" in session))


@app.route("/login")
def login():
    state = secrets.token_hex(16)
    session["spotify_state"] = state

    params = {
        "response_type": "code",
        "client_id": SPOTIFY_CLIENT_ID,
        "scope": "user-read-private",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "state": state,
    }
    return redirect(f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}")


@app.route("/callback")
def callback():
    code = request.args.get("code")

    token_resp = requests.post(SPOTIFY_TOKEN_URL, data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET,
    })

    tokens = token_resp.json()
    session["access_token"] = tokens.get("access_token")

    print("✅ Spotify connected")
    return redirect("/")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/classify", methods=["POST"])
def classify():
    text = request.get_json().get("text", "")

    # ── Embedding ────────────────────────────────────────────────────────
    embedding = st_model.encode([text])
    embedding = np.array(embedding).reshape(1, -1)

    # ── Prediction (safe for all models) ─────────────────────────────────
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(embedding)[0]
        confidence = float(np.max(probs))
        pred_idx = int(np.argmax(probs))
    else:
        pred_idx = int(model.predict(embedding)[0])
        confidence = 0.5

    predicted_label = le.inverse_transform([pred_idx])[0]

    # ── Rule override ────────────────────────────────────────────────────
    override = rule_based_override(text)

    if override and confidence < 0.65:
        mood = override
        source = "rule"
    else:
        mood = predicted_label
        source = "model"

    print(f"{text} → {predicted_label} ({confidence:.2f}) | override={override} → {mood} [{source}]")

    mood_info = MOOD_PARAMS.get(mood, MOOD_PARAMS["neutral"])

    access_token = session.get("access_token")

    if not access_token:
        return jsonify({
            "mood": mood,
            "label": mood_info["label"],
            "emoji": mood_info["emoji"],
            "tracks": [],
            "logged_in": False
        })

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    # ── Spotify search ───────────────────────────────────────────────────
    resp = requests.get(
        "https://api.spotify.com/v1/search",
        headers=headers,
        params={
            "q": mood_info["query"],
            "type": "track",
            "limit": 10,
            "market": "US"
        }
    )

    try:
        data = resp.json()
    except:
        data = {}

    tracks = []

    for t in data.get("tracks", {}).get("items", []):
        tracks.append({
            "name": t["name"],
            "artist": ", ".join(a["name"] for a in t["artists"]),
            "image": t["album"]["images"][0]["url"] if t["album"]["images"] else "",
            "preview": t.get("preview_url"),
            "url": t["external_urls"]["spotify"]
        })

    return jsonify({
        "mood": mood,
        "label": mood_info["label"],
        "emoji": mood_info["emoji"],
        "confidence": round(confidence * 100, 1),
        "tracks": tracks,
        "logged_in": True
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)