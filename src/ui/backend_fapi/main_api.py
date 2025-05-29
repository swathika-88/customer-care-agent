from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
from pathlib import Path
import sys


src_root = Path(__file__).parents[1]  # Go up 2 levels to reach src directory
sys.path.append(str(src_root))

from service.services import process_documents
from service.services import get_retriever
from service.services import ask_question
from service.services import evaluate_rag_pipeline, calculate_metrics

app = FastAPI(title="RAG API Main Endpoint", version="1.0")

# Optional: allow CORS if using a frontend (e.g., Streamlit or React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models

class ChatRequest(BaseModel):
    query: str
    model_name: str = "llama-3.1-8b-instant"

class EvalItem(BaseModel):
    query: str
    reference_answers: str

class EvalRequest(BaseModel):
    questions: List[EvalItem]
    model_name: str = "llama-3.1-8b-instant"


# Helper function to save uploaded files
def save_upload_file(upload_file: UploadFile) -> Path:
    """
    Save uploaded file to the uploads directory
    
    Args:
        upload_file: The uploaded file from FastAPI
        
    Returns:
        Path: The path where the file was saved
    """
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
    
    file_path = upload_dir / upload_file.filename
    
    # Write the uploaded file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path


# Root

@app.get("/")
def root():
    return {"message": "RAG API is running."}


# Upload Endpoint

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to disk
        file_path = save_upload_file(file)
        
        # Reset file pointer to beginning for processing
        file.file.seek(0)
        
        # Process the document (extract text, create chunks, embeddings)
        docs = process_documents(file)
        
        return {"message": f"✅ Uploaded and embedded file: {file.filename}", "file_path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat / RAG Answering Endpoint

@app.post("/chat")
def chat_with_rag(request: ChatRequest):
    try:
        retriever = get_retriever()
        answer = ask_question(request.query, retriever, model_name=request.model_name)
        return {"query": request.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Evaluation Endpoint

@app.post("/evaluate")
def evaluate_rag(request: EvalRequest):
    try:
        retriever = get_retriever()
        questions = [item.model_dump() for item in request.questions]
        results = evaluate_rag_pipeline(retriever, questions, model_name=request.model_name)
        metrics = calculate_metrics(results)
        return {
            "metrics": metrics,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))