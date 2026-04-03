import pandas as pd
import numpy as np
import joblib
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

EMOTION_MAP = {
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "admiration": "joy",
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "fear": "fear",
    "nervousness": "fear",
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
    "disgust": "disgust",
    "neutral": "neutral",
    "caring": "neutral",
    "desire": "neutral",
    "embarrassment": "neutral",
    "approval": "neutral",
}

print("Loading GoEmotions dataset...")
dataset = load_dataset("go_emotions", "simplified")

df = pd.concat(
    [
        dataset["train"].to_pandas(),
        dataset["validation"].to_pandas(),
        dataset["test"].to_pandas(),
    ]
)

label_names = dataset["train"].features["labels"].feature.names


def decode_label(label_list):
    for i in label_list:
        name = label_names[i]
        if name in EMOTION_MAP:
            return EMOTION_MAP[name]
    return None


print("Mapping 28 labels → 7 moods...")
df["mood"] = df["labels"].apply(decode_label)
df = df.dropna(subset=["mood"])
df = df[["text", "mood"]].drop_duplicates()

print(f"\nTotal samples: {len(df):,}")
print("\nSamples per mood:")
print(df["mood"].value_counts().to_string())

print("\nLoading Sentence Transformer...")
st_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Sentence Transformer loaded!")

print("\nEncoding sentences...")
X_embeddings = st_model.encode(
    df["text"].tolist(), show_progress_bar=True, batch_size=128
)
print(f"Embeddings shape: {X_embeddings.shape}")

le = LabelEncoder()
y = le.fit_transform(df["mood"])
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")

# Print label encoding
print("\nLabel encoding:")
for i, name in enumerate(le.classes_):
    print(f"  {i} -> {name}")

# ── Models ────────────────────────────────────────────────────────────────────
# class_weight="balanced" handles imbalance correctly for dense embeddings
# SMOTE was hurting accuracy — synthetic 384-dim vectors aren't semantically valid
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    ),
    "Linear SVM": LinearSVC(
        max_iter=3000,
        class_weight="balanced",
        C=0.1,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    ),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"accuracy": acc, "model": model}
    print(f"  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

best_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]

print("\n" + "=" * 50)
print(f"  BEST MODEL: {best_name}")
print(f"  ACCURACY:   {results[best_name]['accuracy'] * 100:.2f}%")
print("=" * 50)

print("\nAll models ranked:")
ranked = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
for rank, (name, data) in enumerate(ranked, 1):
    marker = " <- best" if name == best_name else ""
    print(f"  {rank}. {name:<25} {data['accuracy'] * 100:.2f}%{marker}")

print("\nClassification Report for best model:")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=class_names))

print(f"\nSaving best model -> best_model.pkl...")
joblib.dump(best_model, "best_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Saved: best_model.pkl")
print("Saved: label_encoder.pkl")

st_model.save("sentence_transformer")
print("Saved: sentence_transformer/")
print("\nAll done! Run: python app.py")