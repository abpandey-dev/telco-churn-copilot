"""Retrieval over the policy knowledge base.

Documents are split into sections ("## ..."), each section becomes a chunk with a stable id
(`retention-offers#offers-for-high-risk-customers`), and chunks are ranked with TF-IDF cosine
similarity. TF-IDF is fully local and deterministic, which keeps tests and CI free; the retriever
is a single class, so swapping in dense embeddings later doesn't touch the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    title: str
    section: str
    text: str


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_chunks() -> list[Chunk]:
    folder = resources.files("churn_copilot") / "knowledge"
    chunks = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        if not path.name.endswith(".md"):
            continue
        doc = path.name.removesuffix(".md")
        text = path.read_text(encoding="utf-8")
        title = re.search(r"^# (.+)$", text, re.M).group(1).strip()
        for section, body in re.findall(r"^## (.+?)\n(.*?)(?=^## |\Z)", text, re.M | re.S):
            chunks.append(Chunk(id=f"{doc}#{_slug(section)}", doc=doc, title=title,
                                section=section.strip(), text=body.strip()))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or load_chunks()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        # Title and section heading are part of what a chunk is "about", so they're indexed too.
        self.matrix = self.vectorizer.fit_transform(
            [f"{c.title}. {c.section}. {c.section}. {c.text}" for c in self.chunks])

    def search(self, query: str, k: int = 4) -> list[dict]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = sorted(zip(scores, self.chunks), key=lambda t: -t[0])[:k]
        return [{"id": c.id, "document": c.title, "section": c.section, "text": c.text,
                 "score": round(float(s), 3)} for s, c in ranked if s > 0]
