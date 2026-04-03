# 🎧 MoodTunes — AI-Powered Mood-Based Music Recommender

MoodTunes is a full-stack AI application that detects a user’s emotional state from natural language input and recommends music in real-time using Spotify.

It combines **NLP (Sentence Transformers)** with a **hybrid ML + rule-based system** to deliver accurate, human-like mood detection and personalized music recommendations.

---

## 🚀 Demo

![MoodTunes Demo](./demo.png)

> 💡 The system detects mood from text and instantly generates a playable playlist using Spotify.

---

## ✨ Features

* 🧠 **Hybrid Emotion Detection**

  * ML model (Sentence Transformers + classifier)
  * Rule-based overrides for real-world accuracy
  * Confidence-aware decision system

* 🎵 **Smart Music Recommendations**

  * Spotify API integration
  * Mood → search query mapping
  * Fallback for tracks without preview

* ▶️ **Interactive Music Player**

  * In-app 30s preview playback
  * Auto-play first recommended track
  * Opens Spotify if preview unavailable

* 💬 **Chat-Based Interface**

  * Conversational UI
  * Real-time mood feedback

* 🔐 **Spotify OAuth Integration**

  * Secure login
  * Access to Spotify track data

---

## 🏗️ Tech Stack

### 🔹 Backend

* Python
* Flask
* Sentence Transformers (`all-MiniLM-L6-v2`)
* Scikit-learn (LinearSVC / Logistic Regression / Random Forest)

### 🔹 Frontend

* HTML, CSS, Vanilla JavaScript
* Responsive UI with custom player

### 🔹 APIs

* Spotify Web API

---

## 🧠 How It Works

1. User inputs text
   👉 *"I feel restless and pumped"*

2. Text is converted into embeddings using Sentence Transformers

3. Model predicts emotion

   * Rule-based override fixes weak predictions

4. Final mood is selected using confidence heuristics

5. Mood is mapped to a music query

6. Spotify API fetches tracks

7. Tracks are displayed + playable in UI

---

## 🔥 Hybrid AI System

Instead of relying only on ML, MoodTunes uses:

```text
Rules + ML + Confidence Heuristics
```

### Why this matters:

| Input       | ML Only        | Hybrid System |
| ----------- | -------------- | ------------- |
| "motivated" | neutral ❌      | joy ✅         |
| "stressed"  | neutral ❌      | fear ✅        |
| "angry"     | inconsistent ❌ | anger ✅       |

---

## ⚙️ Installation & Setup

### 1. Clone repo

```bash
git clone https://github.com/your-username/moodtunes.git
cd moodtunes
```

### 2. Create environment

```bash
uv venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Setup `.env`

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:5000/callback
SECRET_KEY=your_secret
```

---

### 5. Run app

```bash
uv run app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 🔐 Spotify Setup

1. Go to https://developer.spotify.com/dashboard
2. Create an app
3. Add redirect URI:

```
http://127.0.0.1:5000/callback
```

---

## 📂 Project Structure

```
moodtunes/
│── app.py
│── best_model.pkl
│── label_encoder.pkl
│── sentence_transformer/
│── templates/
│    └── index.html
│── static/
│── requirements.txt
│── README.md
│── demo.png
```

---

## ⚡ Key Improvements

* Fixed Spotify `/recommendations` API → switched to `/search`
* Handled missing preview URLs
* Solved model misclassification using hybrid system
* Added robust frontend error handling
* Fixed LinearSVC probability issue

---

## 📊 Model Details

* Dataset: GoEmotions (Google)
* Embedding Model: `all-MiniLM-L6-v2`
* Classifier: Best performing (LinearSVC)
* Accuracy: ~55% (improved via hybrid system)

---

## 🚀 Future Improvements

* 🎯 Audio feature-based recommendations (valence, energy)
* 📊 Confidence visualization in UI
* 🧠 Memory-based personalization
* ☁️ Deployment (Render / Vercel)
* 📱 Mobile optimization

---

## 💼 Resume Bullet

> Built a hybrid AI-powered music recommendation system using NLP (Sentence Transformers), rule-based heuristics, and Spotify API, enabling real-time mood detection and personalized playlist generation.

---

## 🤝 Contributing

Feel free to fork and improve!

---

## 📜 License

MIT License

---

## 👨‍💻 Author

**Shivansh Shukla**

---

⭐ If you like this project, give it a star!
