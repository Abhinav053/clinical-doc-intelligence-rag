from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.api import deps
from app.core.config import get_settings
from app.models.schemas import IngestResponse

UPLOAD_FILE = File(...)

router = APIRouter(tags=['documents'])


@router.post('/documents/ingest', response_model=IngestResponse)
async def ingest_document(file: UploadFile = UPLOAD_FILE) -> IngestResponse:
    settings = get_settings()
    safe_name = Path(file.filename or 'document.pdf').name
    target = settings.upload_dir / safe_name
    target.write_bytes(await file.read())
    document_id, blocks, pages, tables = deps.parser().parse(target)
    chunks = deps.chunker().chunk(blocks)
    deps.store().save_document(document_id, safe_name, pages)
    deps.store().save_blocks(blocks)
    deps.store().save_chunks(chunks)
    all_chunks = deps.store().get_chunks()
    deps.retriever().build(all_chunks)
    return IngestResponse(document_id=document_id, filename=safe_name, pages=pages, blocks=len(blocks), chunks=len(chunks), tables=tables, indexed=True)


@router.get('/documents')
def list_documents():
    return deps.store().list_documents()
