from pydantic import BaseModel


class Chunk(BaseModel):
    law_name: str


class Index(BaseModel):
    law_name: str
