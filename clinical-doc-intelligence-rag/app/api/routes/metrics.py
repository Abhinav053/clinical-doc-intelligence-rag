from fastapi import APIRouter

from app.api import deps

router = APIRouter(tags=['metrics'])


@router.get('/metrics')
def metrics():
    chunks = deps.store().get_chunks()
    docs = deps.store().list_documents()
    return {'documents': len(docs), 'chunks': len(chunks), 'retrieval': 'hybrid_dense_bm25', 'index_ready': bool(chunks)}
