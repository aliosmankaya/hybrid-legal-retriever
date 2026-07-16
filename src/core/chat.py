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
    context = "\n\n".join(r["text"] for r in results)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Sen uzman bir Türk Hukuku yapay zeka asistanısın. Görevin, kullanıcı sorularını "
                "yalnızca sana verilen hukuki bağlamı (kanun maddeleri, yargıtay kararları vb.) kullanarak "
                "doğru, tarafsız ve profesyonel bir şekilde yanıtlamaktır.\n\n"
                "Uyman gereken katı kurallar:\n"
                "1. Yalnızca verilen 'Bağlam' içindeki bilgilere dayanarak cevap ver. Bağlam dışından bilgi veya kanun uydurma.\n"
                "2. Eğer verilen bağlam soruyu cevaplamak için yetersizse, bunu açıkça belirt ve 'Verilen belgelerde bu sorunun cevabı bulunmamaktadır' de.\n"
                "3. Cevap verirken mutlaka bağlamdaki ilgili kanun adı ve madde numaralarına (örn: 2918 sayılı KTK Madde 5) atıf yap.\n"
                "4. Yanıtının sonuna şu yasal uyarıyı ekle: 'Not: Bu yanıt yalnızca bilgilendirme amaçlıdır ve hukuki tavsiye niteliği taşımamaktadır.'",
            ),
            ("user", "Bağlam:\n{context}\n\nSoru: {query}"),
        ]
    )
    chain = prompt | model
    response = chain.invoke({"context": context, "query": query})

    return response.content
