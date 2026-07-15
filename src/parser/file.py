from fastapi import File, UploadFile
from pydantic import BaseModel


class Upload(BaseModel):
    file: UploadFile = File(...)
    law_name: str


class Update(BaseModel):
    current_name: str
    new_name: str


class Delete(BaseModel):
    current_name: str
