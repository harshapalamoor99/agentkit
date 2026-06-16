"""Knowledge / RAG layer — a pluggable retrieval surface the agent can fetch facts from.

The engine is retrieval-agnostic: a :class:`Domain` may expose a :class:`KnowledgeBase`
(via :meth:`Domain.knowledge_base`) and a per-record query (via
:meth:`Domain.knowledge_query`). When both are present the ``context`` node retrieves the
top-k snippets BEFORE the LLM call, injects them into the prompt as grounding, and records
them in the decision lineage + telemetry. Domains with no knowledge base are unaffected.

A dependency-free :class:`InMemoryKnowledgeBase` (pure-python TF-IDF cosine + lexical
overlap) ships as the default so RAG works locally with zero infra. Production backends
(vector DBs, managed retrieval services) plug in behind the same :class:`KnowledgeBase`
interface and are selected via the registry / ``KNOWLEDGE_BACKEND`` without code changes.
"""
from __future__ import annotations

import json
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class KnowledgeSnippet:
    """One retrieved unit of knowledge."""

    id: str
    text: str
    score: float = 0.0
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "score": round(self.score, 4),
                "source": self.source, "metadata": self.metadata}


@dataclass
class KnowledgeDoc:
    id: str
    text: str
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeBase(ABC):
    """Retrieval interface. Implement :meth:`retrieve` for your backend."""

    name: str = "base"

    @abstractmethod
    def retrieve(
        self, query: str, *, k: int = 4, where: dict[str, Any] | None = None
    ) -> list[KnowledgeSnippet]:
        """Return up to ``k`` snippets most relevant to ``query``.

        ``where`` is an optional exact-match metadata filter (e.g. tenant isolation).
        """


class InMemoryKnowledgeBase(KnowledgeBase):
    """Pure-python TF-IDF retriever — zero dependencies, good enough for dev/CI/demos.

    Scoring blends cosine similarity over TF-IDF vectors with a lexical-overlap bonus so
    short keyword queries still match. Deterministic; no network.
    """

    def __init__(self, docs: list[KnowledgeDoc] | None = None, name: str = "in_memory"):
        self.name = name
        self._docs: list[KnowledgeDoc] = []
        self._idf: dict[str, float] = {}
        self._doc_vecs: list[dict[str, float]] = []
        if docs:
            self.add(docs)

    # --- ingestion ---
    def add(self, docs: list[KnowledgeDoc]) -> None:
        self._docs.extend(docs)
        self._reindex()

    @classmethod
    def from_jsonl(cls, path: str | Path, *, name: str | None = None) -> "InMemoryKnowledgeBase":
        """Load docs from a JSONL file. Each line: {id?, text, source?, metadata?}."""
        p = Path(path)
        docs: list[KnowledgeDoc] = []
        if p.exists():
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                text = rec.get("text") or rec.get("content") or ""
                if not text:
                    continue
                docs.append(KnowledgeDoc(
                    id=str(rec.get("id", f"{p.stem}:{i}")),
                    text=str(text),
                    source=rec.get("source") or str(p.name),
                    metadata=rec.get("metadata") or {}))
        return cls(docs, name=name or p.stem)

    def _reindex(self) -> None:
        n = len(self._docs)
        df: dict[str, int] = {}
        doc_tokens: list[list[str]] = []
        for d in self._docs:
            toks = _tokenize(d.text)
            doc_tokens.append(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._doc_vecs = [self._vectorize(toks) for toks in doc_tokens]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        inv = 1.0 / len(tokens)
        return {t: (c * inv) * self._idf.get(t, 1.0) for t, c in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def retrieve(
        self, query: str, *, k: int = 4, where: dict[str, Any] | None = None
    ) -> list[KnowledgeSnippet]:
        qtoks = _tokenize(query)
        if not qtoks or not self._docs:
            return []
        qvec = self._vectorize(qtoks)
        qset = set(qtoks)
        scored: list[KnowledgeSnippet] = []
        for doc, dvec in zip(self._docs, self._doc_vecs):
            if where and any(doc.metadata.get(mk) != mv for mk, mv in where.items()):
                continue
            cos = self._cosine(qvec, dvec)
            overlap = len(qset & set(dvec)) / len(qset) if qset else 0.0
            score = 0.75 * cos + 0.25 * overlap
            if score > 0:
                scored.append(KnowledgeSnippet(
                    id=doc.id, text=doc.text, score=score,
                    source=doc.source, metadata=doc.metadata))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

_REGISTRY: dict[str, KnowledgeBase] = {}


def register_knowledge_base(kb: KnowledgeBase, *, name: str | None = None) -> KnowledgeBase:
    _REGISTRY[name or kb.name] = kb
    return kb


def get_knowledge_base(name: str) -> KnowledgeBase | None:
    return _REGISTRY.get(name)


def available_knowledge_bases() -> list[str]:
    return sorted(_REGISTRY)


def render_snippets(snippets: list[KnowledgeSnippet], *, header: str | None = None) -> str:
    """Default prompt rendering for retrieved knowledge (domains may override)."""
    if not snippets:
        return ""
    head = header or ("Relevant knowledge (retrieved; use it to ground your answer, "
                      "do not contradict it):")
    lines = [head]
    for i, s in enumerate(snippets, 1):
        src = f" [source: {s.source}]" if s.source else ""
        lines.append(f"{i}.{src} {s.text.strip()}")
    return "\n".join(lines)


def build_in_memory_kb_from_env(env_var: str, *, name: str) -> InMemoryKnowledgeBase | None:
    """Convenience: build + register an in-memory KB from a JSONL path in ``env_var``."""
    path = os.getenv(env_var)
    if not path:
        return None
    kb = InMemoryKnowledgeBase.from_jsonl(path, name=name)
    register_knowledge_base(kb, name=name)
    return kb
