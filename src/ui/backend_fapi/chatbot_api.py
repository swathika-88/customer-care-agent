from fastapi import APIRouter, Request, Form
from service.services import ask_question

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

@router.post("/")
async def chat_query(request: Request, query: str = Form(...)):
    generator = request.app.state.generator  # Retrieve from shared app state

    try:
        result = ask_question(generator, query)
        return result
    except Exception as e:
        return {"error": str(e)}
