import os
import numpy as np
from flask import Flask, request, jsonify
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

def get_embedding(text):
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return np.array(response["embedding"])

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * norm(b)))

def grade(score):
    if score < 0.30:
        return 1
    elif score < 0.60:
        return 2
    else:
        return 3

@app.route("/compare", methods=["POST"])
def compare():
    data = request.json
    e1 = get_embedding(data["sentence1"])
    e2 = get_embedding(data["sentence2"])
    sim = cosine_similarity(e1, e2)
    return jsonify({
        "similarity": round(sim, 3),
        "grade": grade(sim)
    })
