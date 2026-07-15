from pydantic import BaseModel


class Chunk(BaseModel):
    file_name: str
    law_name: str


class Index(BaseModel):
    law_name: str
