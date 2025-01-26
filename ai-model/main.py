from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer
import json

# Load your fine-tuned model
model = SentenceTransformer("fine_tuned_faq_model")

# Build question list & answer list
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)
faq_items = data["dataset1"]  # a list of {id, category, question, answer}
questions = [item["question"] for item in faq_items]
answers = [item["answer"] for item in faq_items]

# Compute embeddings for all questions
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define a Pydantic model for the request body
class Query(BaseModel):
    query: str

# Define the endpoint to handle user queries
@app.post("/query/")
async def query_faq(user_query: Query):
    query_text = user_query.query


    # 1) Encode the incoming user question
    user_embedding = model.encode([query_text], convert_to_tensor=True)

    # 2) Compute cosine similarities
    cos_scores = torch.nn.functional.cosine_similarity(question_embeddings, user_embedding[0], dim=1)

    # 3) Find the question with the highest similarity
    top_idx = torch.argmax(cos_scores).item()
    best_score = cos_scores[top_idx].item()

    # Apply the threshold
    if best_score < 0.3:
        return {
            "query": query_text,
            "matched_question": "No match found",
            "answer": "Je suis là pour vous aider. Pouvez-vous reformuler votre question ?",
            "similarity_score": None
        }

    best_match_question = questions[top_idx]
    best_answer = answers[top_idx]

    return {
        "query": query_text,
        "matched_question": best_match_question,
        "answer": best_answer,
        "similarity_score": best_score
    }

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)