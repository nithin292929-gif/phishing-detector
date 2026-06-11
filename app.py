from flask import Flask, render_template, request, jsonify
from model import train_model, predict_email
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    try:
        results = train_model()
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No email text provided"}), 400

    result = predict_email(text)
    return jsonify(result)

@app.route("/model-status")
def model_status():
    trained = os.path.exists("model.pkl")
    return jsonify({"trained": trained})

if __name__ == "__main__":
    app.run(debug=True)