from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..core.chat import invoke
from ..parser.chat import Chat

router = APIRouter()


@router.post("/chat")
def chat_service(params: Chat):
    response = invoke(query=params.query, law_name=params.law_name, limit=params.limit)
    return JSONResponse(content=response, status_code=200)
