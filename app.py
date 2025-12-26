import torch
from numpy import dot
from numpy.linalg import norm
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)

app = FastAPI(
    title="Sentence Correlation API",
    description="Grades semantic correlation between sentences",
    version="1.0"
)

# 🔹 Lazy-loaded model
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


class SentencePair(BaseModel):
    sentence1: str
    sentence2: str


def cosine_similarity(a, b):
    return float(dot(a, b) / (norm(a) * norm(b)))


def grade_similarity(score: float):
    if score < 0.30:
        return 1
    elif score < 0.60:
        return 2
    else:
        return 3


@app.post("/compare")
def compare_sentences(data: SentencePair):
    model = get_model()
    embeddings = model.encode(
        [data.sentence1, data.sentence2],
        convert_to_numpy=True
    )

    similarity = cosine_similarity(embeddings[0], embeddings[1])

    return {
        "sentence_1": data.sentence1,
        "sentence_2": data.sentence2,
        "similarity_score": round(similarity, 3),
        "correlation_grade": grade_similarity(similarity)
    }


@app.get("/test-dummy")
def test_dummy_data():
    model = get_model()

    dummy_tests = [
        {
            "sentence1": "Solve matrix operations and understand vectors",
            "sentence2": "Linear algebra concepts involving matrices and vectors"
        },
        {
            "sentence1": "Analyze computing problems using algorithms",
            "sentence2": "Solve matrix operations and work with complex numbers"
        },
        {
            "sentence1": "Develop mobile applications using Android Studio",
            "sentence2": "Understand matrices and linear dependence of vectors"
        }
    ]

    results = []

    for test in dummy_tests:
        embeddings = model.encode(
            [test["sentence1"], test["sentence2"]],
            convert_to_numpy=True
        )

        similarity = cosine_similarity(embeddings[0], embeddings[1])

        results.append({
            "sentence_1": test["sentence1"],
            "sentence_2": test["sentence2"],
            "similarity_score": round(similarity, 3),
            "correlation_grade": grade_similarity(similarity)
        })

    return {
        "message": "Dummy test results",
        "results": results
    }

    return {
        "message": "Dummy test results",
        "results": results
    }
