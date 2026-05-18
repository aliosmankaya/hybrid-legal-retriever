from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        if not chunks:
            return []

        # Query + her chunk'ı çift olarak ver
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Skorları hesapla — tek batch, CPU'da ~1-2sn
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Skor ile chunk'ı eşleştir, sırala
        scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        # Top-k chunk'ı reranker skoru ile döndür
        return [
            {**chunk, "reranker_score": float(score)}
            for chunk, score in scored_chunks[:top_k]
        ]
