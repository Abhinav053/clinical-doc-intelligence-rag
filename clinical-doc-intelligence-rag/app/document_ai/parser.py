from __future__ import annotations

import hashlib
from pathlib import Path

import fitz
import pdfplumber
import pytesseract
from PIL import Image

from app.core.config import Settings
from app.models.schemas import EvidenceBlock


class ClinicalDocumentParser:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def parse(self, file_path: Path) -> tuple[str, list[EvidenceBlock], int, int]:
        document_id = hashlib.sha1(file_path.read_bytes()).hexdigest()[:16]
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            return document_id, *self._parse_pdf(file_path, document_id)
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image) if self.settings.enable_ocr else ''
        block = EvidenceBlock(document_id=document_id, page_number=1, block_id=f'{document_id}_p1_ocr_1', block_type='ocr_text', text=text, confidence=None)
        return document_id, [block], 1, 0

    def _parse_pdf(self, file_path: Path, document_id: str) -> tuple[list[EvidenceBlock], int, int]:
        blocks: list[EvidenceBlock] = []
        table_count = 0
        doc = fitz.open(file_path)
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text('text').strip()
            if text:
                blocks.append(EvidenceBlock(document_id=document_id, page_number=page_index, block_id=f'{document_id}_p{page_index}_text_1', block_type='pdf_text', text=text))
            if self.settings.enable_ocr and len(text) < self.settings.min_pdf_text_chars:
                pix = page.get_pixmap(dpi=180)
                image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(image).strip()
                if ocr_text:
                    blocks.append(EvidenceBlock(document_id=document_id, page_number=page_index, block_id=f'{document_id}_p{page_index}_ocr_1', block_type='ocr_text', text=ocr_text))
        with pdfplumber.open(file_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                for table_index, table in enumerate(page.extract_tables() or [], start=1):
                    rows = [' | '.join([cell or '' for cell in row]) for row in table if row]
                    table_text = '\n'.join(rows).strip()
                    if table_text:
                        table_count += 1
                        blocks.append(EvidenceBlock(document_id=document_id, page_number=page_index, block_id=f'{document_id}_p{page_index}_table_{table_index}', block_type='table', text=table_text))
        return blocks, len(doc), table_count
