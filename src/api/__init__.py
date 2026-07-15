from fastapi import FastAPI

from .file import router as file_router
from .index import router as index_router

app = FastAPI()


app.include_router(router=file_router, prefix="/file", tags=["File"])
app.include_router(router=index_router, prefix="/index", tags=["Index"])
