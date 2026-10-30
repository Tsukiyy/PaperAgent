"""基于 FAISS 的本地证据文本向量索引。"""

import hashlib
import json
import re
from pathlib import Path

import faiss
import numpy as np

from paper_agent.models import DocumentChunk

TOKEN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class FaissVectorIndex:
    """用稳定哈希嵌入建立可离线运行的余弦相似度索引。"""

    def __init__(self, directory: Path, *, dimension: int = 384) -> None:
        self.directory = directory
        self.dimension = dimension
        self._index: faiss.Index | None = None
        self._metadata: list[dict[str, object]] = []

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype="float32")
        for token in TOKEN.findall(text.casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % self.dimension] += 1.0 if value & 1 else -1.0
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector

    def build(self, records: list[tuple[str, DocumentChunk]]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._metadata = [
            {
                "paper_id": paper_id,
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "text": chunk.text,
            }
            for paper_id, chunk in records
        ]
        base = faiss.IndexFlatIP(self.dimension)
        self._index = base
        if records:
            vectors = np.vstack(
                [self._embed(f"{chunk.section_title}\n{chunk.text}") for _, chunk in records]
            )
            self._index.add(vectors)
        faiss.write_index(self._index, str(self.directory / "chunks.faiss"))
        (self.directory / "chunks.metadata.json").write_text(
            json.dumps(self._metadata, ensure_ascii=False),
            encoding="utf-8",
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 12,
        paper_id: str | None = None,
    ) -> tuple[str, ...]:
        if self._index is None or self._index.ntotal == 0:
            return ()
        # 指定论文时必须先拿到足够大的候选集再过滤。旧实现只搜索全局前
        # limit * 4 条，多篇论文共用索引时，目标论文经常只剩下 1～4 条结果。
        expanded_limit = (
            self._index.ntotal
            if paper_id
            else min(self._index.ntotal, max(limit * 4, limit))
        )
        _, indexes = self._index.search(self._embed(query).reshape(1, -1), expanded_limit)
        matches: list[str] = []
        for index in indexes[0]:
            if index < 0:
                continue
            metadata = self._metadata[int(index)]
            if paper_id and metadata["paper_id"] != paper_id:
                continue
            matches.append(str(metadata["chunk_id"]))
            if len(matches) >= limit:
                break
        return tuple(matches)
