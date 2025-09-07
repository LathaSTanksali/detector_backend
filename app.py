from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import numpy as np
from collections import Counter
import os

app = Flask(__name__)
CORS(app)

# Load model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Stats counter
stats = Counter({"phishing": 0, "legitimate": 0})

@app.route("/api/classify", methods=["POST"])
def classify_url():
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Transform raw URL
    features = vectorizer.transform([url])
    prediction = model.predict(features)[0]

    # Update stats
    if prediction == 1:
        stats["phishing"] += 1
    else:
        stats["legitimate"] += 1

    # --- Explainability: find top contributing features ---
    if hasattr(model, "coef_"):
        feature_names = np.array(vectorizer.get_feature_names_out())
        coefficients = model.coef_[0]

        # Multiply feature weights with TF-IDF values
        contribution = features.toarray()[0] * coefficients

        # Get top 5 features contributing most
        top_indices = np.argsort(contribution)[-5:][::-1]
        top_features = feature_names[top_indices].tolist()
    else:
        top_features = []

    return jsonify({
        "prediction": int(prediction),
        "top_features": top_features
    })

@app.route("/api/stats", methods=["GET"])
def get_stats():
    total = stats["phishing"] + stats["legitimate"]
    phishing_percent = (stats["phishing"] / total * 100) if total > 0 else 0
    legitimate_percent = (stats["legitimate"] / total * 100) if total > 0 else 0

    return jsonify({
        "phishing": stats["phishing"],
        "legitimate": stats["legitimate"],
        "phishing_percent": phishing_percent,
        "legitimate_percent": legitimate_percent
    })

if __name__ == "__main__":
    # Render provides PORT as env variable, default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
