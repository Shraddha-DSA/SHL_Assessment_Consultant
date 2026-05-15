from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from .engine import SHLConsultantEngine

app = FastAPI(title="SHL Assessment Recommender API")

# Initialize Engine
engine = SHLConsultantEngine()

# --- Schemas ---

class Message(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = []
    end_of_conversation: bool = False

# --- Endpoints ---

@app.get("/")
async def root():

    return {
        "message": "SHL Assessment Recommender API is running successfully.",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
async def health_check():
    """Readiness endpoint for automated evaluator."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    State-less chat endpoint that takes full history 
    and returns reply + structured recommendations.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Message history cannot be empty")
    
    # Convert Pydantic models to dicts for engine
    history = [m.dict() for m in request.messages]
    
    result = engine.generate_response(history)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)