import os
import math
from flask import Flask, request, jsonify
import google.generativeai as genai

# Gemini API key from Vercel env
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

# ---------- UTILITIES ----------

def get_embedding(text):
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return response["embedding"]  # list of floats


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def grade(score):
    if score < 0.30:
        return 1
    elif score < 0.60:
        return 2
    else:
        return 3


# ---------- ROUTES ----------

@app.route("/")
def home():
    return jsonify({"status": "API running"})


@app.route("/compare", methods=["POST"])
def compare():
    data = request.json

    e1 = get_embedding(data["sentence1"])
    e2 = get_embedding(data["sentence2"])

    sim = cosine_similarity(e1, e2)

    return jsonify({
        "similarity_score": round(sim, 3),
        "correlation_grade": grade(sim)
    })
