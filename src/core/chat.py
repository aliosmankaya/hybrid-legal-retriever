import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter

from .hybrid_retriever import retriever


def invoke(query: str, law_name: str, limit: int):
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = "nvidia/nemotron-3-ultra-550b-a55b:free"
    model = ChatOpenRouter(model=model_name, api_key=api_key)

    results = retriever(query=query, law_name=law_name, limit=limit)
    context = "\n\n".join(r.payload["text"] for r in results)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Sen bir Türk Hukuku asistanısın."),
            ("user", f"Bağlam:\n{context}\n\nSoru: {query}"),
        ]
    )
    chain = prompt | model
    response = chain.invoke({"context": context, "query": query})

    return response.content
