from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .rag import answer_question

app = FastAPI(title="RAG Chatbot (Groq + Chroma)")

# Serve static files (CSS/JS etc) from /static
app.mount("/static", StaticFiles(directory="static"), name="static")


# Root: serve the chat UI
@app.get("/")
def root():
    return FileResponse("static/index.html")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = answer_question(req.question)
    return ChatResponse(**result)
