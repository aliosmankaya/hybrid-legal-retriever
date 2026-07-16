import os
import pickle

from rank_bm25 import BM25Okapi

from .helper import tokenize_tr


def create_bm25_index(chunks: list[dict]):
    tokenized_coprpus = [tokenize_tr(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_coprpus)


def save_bm25_index(bm25: BM25Okapi, chunks: list[dict], law_name: str):
    path = f"{os.getcwd()}/data/{law_name}/bm25_index.pkl"
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)


def load_bm25_index(law_name: str):
    path = f"{os.getcwd()}/data/{law_name}/bm25_index.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def retriever(query: str, law_name: str, limit: int = 20):
    tokenized_query = tokenize_tr(query)
    bm25, chunks = load_bm25_index(law_name=law_name)
    scores = bm25.get_scores(tokenized_query)
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:limit]
    return [(chunks[idx]["chunk_id"], float(score)) for idx, score in scored]
