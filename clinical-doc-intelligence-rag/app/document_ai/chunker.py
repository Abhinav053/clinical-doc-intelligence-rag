from __future__ import annotations

import hashlib

from app.models.schemas import Chunk, EvidenceBlock


class ProvenanceChunker:
    def __init__(self, chunk_size: int, overlap: int) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, blocks: list[EvidenceBlock]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for block in blocks:
            text = ' '.join(block.text.split())
            if not text:
                continue
            step = max(1, self.chunk_size - self.overlap)
            for start in range(0, len(text), step):
                part = text[start:start + self.chunk_size]
                if len(part.strip()) < 20:
                    continue
                digest = hashlib.sha1(f'{block.block_id}:{start}:{part}'.encode()).hexdigest()[:12]
                chunks.append(Chunk(chunk_id=f'{block.document_id}_{digest}', document_id=block.document_id, page_numbers=[block.page_number], text=part, source_blocks=[block.block_id], source_type=block.block_type))
        return chunks
