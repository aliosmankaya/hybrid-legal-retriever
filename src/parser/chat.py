from pydantic import BaseModel


class Chat(BaseModel):
    query: str
    law_name: str
    limit: int = 20
