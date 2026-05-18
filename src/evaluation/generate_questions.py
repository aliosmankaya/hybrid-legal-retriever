import requests


def generate_questions_for_chunk(chunk: dict, n: int = 2) -> list[str]:
    prompt = f"""Aşağıdaki Türk hukuku metninden {n} adet soru üret.
    Kurallar:
    - Sorular SADECE bu metne bakılarak cevaplanabilir olmalı
    - Her soru ayrı satırda olmalı
    - Soru dışında hiçbir şey yazma
    
    Metin ({chunk['metadata']['kanun']} - Madde {chunk['metadata']['madde_no']}):
    {chunk['text'][:800]}"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:3b", "prompt": prompt, "stream": False},
    )
    raw = response.json()["response"].strip()
    return [q.strip() for q in raw.split("\n") if q.strip()]
