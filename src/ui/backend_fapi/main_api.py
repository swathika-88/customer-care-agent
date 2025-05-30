from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
from pathlib import Path
import sys
import os

src_root = Path(__file__).parents[1]  # Go up 2 levels to reach src directory
sys.path.append(str(src_root))

from service.services import process_documents
from service.services import get_retriever
from service.services import ask_question
from service.services import evaluate_rag_pipeline, calculate_metrics

app = FastAPI(
    title="RAG API Main Endpoint", 
    version="1.0",
    description="Complete RAG pipeline API for document processing, Q&A, and evaluation"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class ChatRequest(BaseModel):
    query: str
    # model_name: str = "llama-3.1-8b-instant"

class EvalItem(BaseModel):
    query: str
    reference_answers: str

class EvalRequest(BaseModel):
    questions: List[EvalItem]
    model_name: str = "llama-3.1-8b-instant"

class MultiUploadRequest(BaseModel):
    files: List[UploadFile]

# Helper function to save uploaded files
def save_upload_file(upload_file: UploadFile, upload_dir: Path = Path("uploads")) -> Path:
    """Save uploaded file to the uploads directory"""
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / upload_file.filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path

# Root Endpoint
@app.get("/")
def root():
    return {
        "message": "RAG API is running",
        "version": "1.0",
        "endpoints": {
            "/upload": "POST - Upload single document",
            "/upload-multiple": "POST - Upload multiple documents",
            "/chat": "POST - Ask questions to RAG system",
            "/evaluate": "POST - Evaluate RAG performance",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

# Multiple Files Upload Endpoint
@app.post("/upload-multiple")
# def upload_multiple_files(files: List[UploadFile] = File(...)):
def upload_multiple_files(files: UploadFile = File(...)):
    """Upload and process multiple documents"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        uploaded_files = []
        processing_results = []
        
        
        # for file in files:
        #     # Validate file type
        #     if not file.filename.lower().endswith('.pdf'):
        #         raise HTTPException(
        #             status_code=400, 
        #             detail=f"File {file.filename} is not a PDF. Only PDF files are supported"
        #         )
            
        #     # Save file
        #     file_path = save_upload_file(file)
        #     uploaded_files.append(str(file_path))
        #     print( 100 * '-')
            
        #     # Reset file pointer and process
        #     file.file.seek(0)
        #     result = process_documents(file)
        #     processing_results.append(result)
        
        file = files
        if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a PDF. Only PDF files are supported"
                )
            
            # Save file
        file_path = save_upload_file(file)
        uploaded_files.append(str(file_path))
        
        
        # Reset file pointer and process
        file.file.seek(0)
        result = process_documents(file)
        processing_results.append(result)
        return {
            "message": f"✅ Uploaded and processed  files",
            # "files": [f.filename for f in files],
            "files": file,
            "file_paths": uploaded_files,
            "processing_results": processing_results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multiple upload failed: {str(e)}")

# Chat / RAG Answering Endpoint
@app.post("/chat")
def chat_with_rag(request: ChatRequest):
    """Ask questions to the RAG system"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        print("started geting the retriver")
        retriever = get_retriever()
        # answer = ask_question(request.query, retriever, model_name=request.model_name)
        print("retriver is done")
        
        answer = ask_question(query = request.query,generator = retriever)
        
        return {
            "query": request.query,
            "answer": answer,
            "model_used": request.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# Evaluation Endpoint
@app.post("/evaluate")
def evaluate_rag(request: EvalRequest):
    """Evaluate RAG pipeline performance"""
    try:
        if not request.questions:
            raise HTTPException(status_code=400, detail="No evaluation questions provided")
        
        retriever = get_retriever()
        questions = [item.model_dump() for item in request.questions]
        results = evaluate_rag_pipeline(retriever, questions, model_name=request.model_name)
        metrics = calculate_metrics(results)
        
        return {
            "metrics": metrics,
            "results": results,
            "model_used": request.model_name,
            "total_questions": len(questions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# Get Upload Status
@app.get("/uploads")
def list_uploads():
    """List all uploaded files"""
    try:
        uploads_dir = Path("uploads")
        if not uploads_dir.exists():
            return {"uploaded_files": [], "count": 0}
        
        files = [f.name for f in uploads_dir.iterdir() if f.is_file()]
        return {
            "uploaded_files": files,
            "count": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list uploads: {str(e)}")

