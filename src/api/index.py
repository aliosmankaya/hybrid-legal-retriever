from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..core.index import chunking, indexing
from ..parser.index import Chunk, Index

router = APIRouter()


@router.post("/chunking")
def chunking_service(params: Chunk):
    chunking(file_name=params.file_name, law_name=params.law_name)
    return JSONResponse(content="Chunking successfully", status_code=200)


@router.post("/indexing")
def indexing_service(params: Index):
    indexing(law_name=params.law_name)
    return JSONResponse(content="Indexed successfully", status_code=200)
