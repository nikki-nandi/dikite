import pickle
import numpy as np
import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------
# App setup
# --------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lovable needs this
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------
# Load spaCy
# --------------------------------
nlp = spacy.load("en_core_web_sm")

def preprocess(text: str) -> str:
    doc = nlp(text.lower())
    return " ".join(
        token.lemma_ for token in doc if token.is_alpha and not token.is_stop
    )

# --------------------------------
# Load embedding model
# --------------------------------
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------
# Load chatbot data
# --------------------------------
with open("dikite_chatbot_model.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
answers = data["answers"]
embeddings = np.array(data["embeddings"])

# --------------------------------
# API schema
# --------------------------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# --------------------------------
# Chat endpoint
# --------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_input = preprocess(req.message)

    if not user_input.strip():
        return {"reply": "Please ask a valid question."}

    # Encode user message
    user_embedding = bert_model.encode([user_input])

    # Compute similarity
    similarities = cosine_similarity(user_embedding, embeddings)[0]

    # Best match
    best_index = int(np.argmax(similarities))
    best_score = similarities[best_index]

    # Optional confidence threshold
    if best_score < 0.4:
        return {"reply": "Sorry, I don't understand that question."}

    return {"reply": answers[best_index]}

# --------------------------------
# Health check
# --------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
