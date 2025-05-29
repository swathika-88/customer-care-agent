from fastapi import APIRouter, UploadFile, File
from typing import List
from service import process_documents

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        result = process_documents(files)
        return result
    except Exception as e:
        return {"error": str(e)}
