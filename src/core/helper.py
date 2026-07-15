import json
import os
import re


def tokenize_tr(text: str) -> list[str]:
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = re.sub(r"[^a-zçğıöşü0-9\s]", " ", text)
    return text.split()


def load_chunks(law_name: str):
    path = os.get_cwd() + "/data/" + law_name + "/chunks/chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
