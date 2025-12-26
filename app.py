import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from reportlab.platypus import SimpleDocTemplate, Table
import google.generativeai as genai

# 🔐 Gemini API key from Vercel Environment Variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

# ---------- UTIL FUNCTIONS ----------

def get_embedding(text: str):
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return np.array(response["embedding"])


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def grade(score):
    if score < 0.30:
        return 1
    elif score < 0.60:
        return 2
    else:
        return 3


# ---------- API ENDPOINTS ----------

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


@app.route("/upload", methods=["POST"])
def upload_files():
    co_file = request.files["co"]
    po_file = request.files["po"]

    co_list = pd.read_excel(co_file).iloc[:, 0].dropna().tolist()
    po_list = pd.read_excel(po_file).iloc[:, 0].dropna().tolist()

    table_data = [["CO Statement", "PO Statement", "Similarity", "Grade"]]

    for co in co_list:
        for po in po_list:
            sim = cosine_similarity(get_embedding(co), get_embedding(po))
            table_data.append([co, po, round(sim, 3), grade(sim)])

    pdf_path = "co_po_report.pdf"
    doc = SimpleDocTemplate(pdf_path)
    doc.build([Table(table_data)])

    return send_file(pdf_path, as_attachment=True)


@app.route("/")
def home():
    return jsonify({"status": "CO-PO Correlation API is running"})
