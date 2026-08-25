from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app.api import deps
from app.core.config import get_settings
from app.safety.phi_redactor import redact_phi
from app.safety.query_classifier import classify_query

st.set_page_config(page_title='Clinical Document Intelligence RAG', layout='wide')
st.title('Clinical Document Intelligence RAG')
st.caption('OCR/layout-aware, citation-first clinical evidence RAG demo for clinicians. Not for diagnosis.')

tab_ingest, tab_query, tab_evidence, tab_eval = st.tabs(['Ingest', 'Ask', 'Evidence', 'Metrics'])

with tab_ingest:
    uploaded = st.file_uploader('Upload PDF/image', type=['pdf', 'png', 'jpg', 'jpeg'])
    if uploaded and st.button('Ingest document'):
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)
        document_id, blocks, pages, tables = deps.parser().parse(tmp_path)
        chunks = deps.chunker().chunk(blocks)
        deps.store().save_document(document_id, uploaded.name, pages)
        deps.store().save_blocks(blocks)
        deps.store().save_chunks(chunks)
        deps.retriever().build(deps.store().get_chunks())
        st.success(f'Indexed {len(chunks)} chunks from {pages} pages, {len(blocks)} blocks, {tables} tables.')

with tab_query:
    question = st.text_area('Clinical evidence question', 'What evidence is available in the uploaded document?')
    if st.button('Retrieve and answer'):
        redacted, phi_flags = redact_phi(question)
        flags = sorted(set(phi_flags + classify_query(redacted)))
        citations = deps.retriever().search(redacted, k=get_settings().rerank_k)
        answer = deps.llm_provider().generate(redacted, citations, flags)
        st.subheader('Answer')
        st.write(answer.answer)
        st.warning(answer.disclaimer)
        st.subheader('Safety flags')
        st.write(answer.safety_flags or 'None')
        st.subheader('Citations')
        for c in answer.citations:
            st.markdown(f'**{c.document_id} pages {c.pages} score={c.score}**')
            st.code(c.quote)

with tab_evidence:
    docs = deps.store().list_documents()
    st.dataframe(pd.DataFrame(docs))
    chunks = deps.store().get_chunks()
    st.write(f'Total chunks: {len(chunks)}')
    if chunks:
        st.dataframe(pd.DataFrame([c.model_dump() for c in chunks[:50]]))

with tab_eval:
    chunks = deps.store().get_chunks()
    st.metric('Indexed chunks', len(chunks))
    st.metric('Indexed documents', len(deps.store().list_documents()))
    st.info('Run `python scripts/run_eval.py` after adding gold chunk IDs to data/eval/golden_dataset.json.')
