import re
import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi


def load_chunks(jsonl_path: str) -> list[dict]:
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize_tr(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized_corpus = [tokenize_tr(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def save_bm25_index(bm25: BM25Okapi, chunks: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"BM25 index kaydedildi: {path}")


def load_bm25_index(path: str) -> tuple[BM25Okapi, list[dict]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def bm25_search(
    query: str, bm25: BM25Okapi, chunks: list[dict], top_k: int = 20
) -> list[tuple[dict, float]]:
    tokenized_query = tokenize_tr(query)
    scores = bm25.get_scores(tokenized_query)
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    return [(chunks[idx], float(score)) for idx, score in scored]
