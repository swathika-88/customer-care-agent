from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from service.services import evaluate_rag_pipeline, calculate_metrics
from service.services import get_retriever
router = APIRouter()

class QuestionItem(BaseModel):
    query: str
    reference_answers: str

class EvaluationRequest(BaseModel):
    questions: List[QuestionItem]
    model_name: str = "llama-3.1-8b-instant"

@router.post("/evaluate")
def evaluate_rag(request: EvaluationRequest):
    try:
        retriever = get_retriever()
        results = evaluate_rag_pipeline(retriever, [q.dict() for q in request.questions], model_name=request.model_name)
        metrics = calculate_metrics(results)
        return {
            "metrics": metrics,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))