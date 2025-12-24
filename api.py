from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="Sentence Correlation API",
    description="Grades semantic correlation between sentences",
    version="1.0"
)

# Load model ONCE (important for performance)
model = SentenceTransformer("all-MiniLM-L6-v2")


class SentencePair(BaseModel):
    sentence1: str
    sentence2: str


def grade_similarity(score: float):
    if score < 0.30:
        return 1
    elif score < 0.60:
        return 2
    else:
        return 3


@app.post("/compare")
def compare_sentences(data: SentencePair):
    embeddings = model.encode([data.sentence1, data.sentence2])
    similarity = cosine_similarity(
        [embeddings[0]], [embeddings[1]]
    )[0][0]

    grade = grade_similarity(similarity)

    return {
        "sentence_1": data.sentence1,
        "sentence_2": data.sentence2,
        "similarity_score": round(similarity, 3),
        "correlation_grade": grade
    }
