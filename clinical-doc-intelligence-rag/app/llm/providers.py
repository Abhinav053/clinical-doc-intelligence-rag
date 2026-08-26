from __future__ import annotations

import httpx
from groq import Groq

from app.core.config import Settings
from app.core.constants import CLINICAL_DISCLAIMER
from app.models.schemas import ClinicalAnswer, EvidenceCitation


class LLMProvider:
    def generate(self, question: str, citations: list[EvidenceCitation], safety_flags: list[str]) -> ClinicalAnswer:
        raise NotImplementedError


class MockProvider(LLMProvider):
    def generate(self, question: str, citations: list[EvidenceCitation], safety_flags: list[str]) -> ClinicalAnswer:
        if not citations:
            return ClinicalAnswer(answer='I could not find enough evidence in the indexed documents.', citations=[], confidence='low', limitations='No relevant chunks were retrieved.', safety_flags=safety_flags, disclaimer=CLINICAL_DISCLAIMER)
        findings = [f'Evidence from document {c.document_id}, page(s) {c.pages}: {c.quote[:120]}' for c in citations[:3]]
        return ClinicalAnswer(answer='Based on the retrieved clinical evidence, review the cited source snippets below. This demo provider avoids unsupported clinical conclusions.', key_findings=findings, citations=citations, confidence='moderate' if len(citations) >= 2 else 'low', limitations='Demo mode uses extractive evidence synthesis; clinician judgment is required.', safety_flags=safety_flags, disclaimer=CLINICAL_DISCLAIMER)


class GroqProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key)

    def generate(self, question: str, citations: list[EvidenceCitation], safety_flags: list[str]) -> ClinicalAnswer:
        context = '\n\n'.join(f'[{i+1}] doc={c.document_id} pages={c.pages} text={c.quote}' for i, c in enumerate(citations))
        prompt = f'''You are a clinician-facing evidence assistant. Do not diagnose or prescribe. Use only context.
Question: {question}
Safety flags: {safety_flags}
Context:\n{context}\nReturn concise findings and limitations.'''
        response = self.client.chat.completions.create(model=self.settings.groq_model, messages=[{'role': 'user', 'content': prompt}], temperature=0.1, max_tokens=700)
        answer = response.choices[0].message.content or ''
        return ClinicalAnswer(answer=answer, key_findings=[], citations=citations, confidence='moderate' if citations else 'low', limitations='Generated from retrieved context only; verify source documents.', safety_flags=safety_flags, disclaimer=CLINICAL_DISCLAIMER)


class OllamaProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, question: str, citations: list[EvidenceCitation], safety_flags: list[str]) -> ClinicalAnswer:
        context = '\n'.join(c.quote for c in citations)
        payload = {'model': self.settings.ollama_model, 'prompt': f'Question: {question}\nContext:\n{context}\nAnswer with citations and limitations.', 'stream': False}
        data = httpx.post(self.settings.ollama_url, json=payload, timeout=60).json()
        return ClinicalAnswer(answer=data.get('response', ''), citations=citations, confidence='moderate' if citations else 'low', limitations='Local Ollama generation; verify evidence.', safety_flags=safety_flags, disclaimer=CLINICAL_DISCLAIMER)


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == 'groq' and settings.groq_api_key:
        return GroqProvider(settings)
    if settings.llm_provider == 'ollama':
        return OllamaProvider(settings)
    return MockProvider()
