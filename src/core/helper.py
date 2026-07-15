import re


def tokenize_tr(text: str) -> list[str]:
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = re.sub(r"[^a-zçğıöşü0-9\s]", " ", text)
    return text.split()
