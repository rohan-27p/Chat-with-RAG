"""
FastAPI application - PDF Chat backend.

Endpoints (API-compatible with the existing TypeScript frontend)
---------------------------------------------------------------
POST   /api/upload                   Upload one or more PDFs; extract, chunk, persist
POST   /api/chat                     Ask a question about an uploaded session
GET    /api/chat/session/{sessionId} Get session metadata
DELETE /api/chat/session/{sessionId} Delete a session
GET    /api/health                   Liveness check
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Allow imports from the backend root (where config.py lives)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (  # noqa: E402
    CORS_ORIGIN,
    LLM_MODEL,
    MAX_PDF_SIZE_MB,
    PORT,
    STORAGE_DIR,
    UPLOAD_DIR,
)
from app.chunker import chunk_pages  # noqa: E402
from app.embeddings import FAISSStore  # noqa: E402
from app.llm import answer_question  # noqa: E402
from app.pdf_processor import extract_pages  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    yield


app = FastAPI(title="PDF Chat API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_session_cache: dict[str, FAISSStore] = {}
_session_info: dict[str, dict] = {}

_INFO_FILE = "info.json"


def _info_path(session_id: str) -> str:
    return os.path.join(STORAGE_DIR, session_id, _INFO_FILE)


def _load_info(session_id: str) -> dict | None:
    path = _info_path(session_id)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_info(session_id: str, info: dict) -> None:
    os.makedirs(os.path.join(STORAGE_DIR, session_id), exist_ok=True)
    with open(_info_path(session_id), "w", encoding="utf-8") as f:
        json.dump(info, f)


def _get_store(session_id: str) -> FAISSStore:
    if session_id not in _session_cache:
        if not FAISSStore.exists(session_id):
            raise HTTPException(
                status_code=404,
                detail="Session not found. Please re-upload your PDF files.",
            )
        _session_cache[session_id] = FAISSStore.load(session_id)
    return _session_cache[session_id]


class ChatRequest(BaseModel):
    sessionId: str
    question: str


class Citation(BaseModel):
    fileName: str
    page: int
    snippet: str


class UploadedFileInfo(BaseModel):
    fileName: str
    pageCount: int


class UploadResponse(BaseModel):
    sessionId: str
    fileName: str
    pageCount: int
    fileCount: int
    files: list[UploadedFileInfo]
    message: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    isOutOfScope: bool


class SessionInfo(BaseModel):
    fileName: str
    pageCount: int
    fileCount: int
    files: list[UploadedFileInfo]
    createdAt: str


def _summarise_session_files(files: list[dict]) -> tuple[str, int]:
    if not files:
        return "document.pdf", 0
    if len(files) == 1:
        return files[0]["fileName"], files[0]["pageCount"]
    return f"{len(files)} files", sum(file["pageCount"] for file in files)


async def _get_uploaded_pdfs(request: Request) -> list[UploadFile]:
    form = await request.form()
    uploads: list[UploadFile] = []

    for field_name in ("pdf", "pdfs"):
        for value in form.getlist(field_name):
            if hasattr(value, "filename") and hasattr(value, "read"):
                uploads.append(value)

    return uploads


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(request: Request):
    uploads = await _get_uploaded_pdfs(request)
    if not uploads:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    session_id = str(uuid.uuid4())
    staged_paths: list[str] = []

    try:
        all_pages: list[dict] = []
        file_summaries: list[dict] = []

        for file_index, pdf in enumerate(uploads, start=1):
            if pdf.content_type not in ("application/pdf", "application/octet-stream"):
                if not (pdf.filename or "").lower().endswith(".pdf"):
                    raise HTTPException(
                        status_code=400,
                        detail="Only PDF files are accepted.",
                    )

            content = await pdf.read()
            max_bytes = MAX_PDF_SIZE_MB * 1024 * 1024
            if len(content) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File '{pdf.filename or f'document-{file_index}.pdf'}' "
                        f"exceeds the {MAX_PDF_SIZE_MB} MB limit."
                    ),
                )

            original_name = pdf.filename or f"document-{file_index}.pdf"
            pdf_path = os.path.join(UPLOAD_DIR, f"{session_id}-{file_index}.pdf")
            staged_paths.append(pdf_path)

            with open(pdf_path, "wb") as fh:
                fh.write(content)

            pages = extract_pages(pdf_path, file_name=original_name)
            all_pages.extend(pages)
            file_summaries.append(
                {
                    "fileName": original_name,
                    "pageCount": len(pages),
                }
            )

        chunks = chunk_pages(all_pages)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any text from the uploaded PDFs.",
            )

        store = FAISSStore(session_id)
        store.build(chunks)
        store.save()

        primary_file_name, total_pages = _summarise_session_files(file_summaries)
        info = {
            "fileName": primary_file_name,
            "pageCount": total_pages,
            "fileCount": len(file_summaries),
            "files": file_summaries,
            "createdAt": datetime.utcnow().isoformat(),
        }
        _save_info(session_id, info)

        _session_cache[session_id] = store
        _session_info[session_id] = info

        return UploadResponse(
            sessionId=session_id,
            fileName=primary_file_name,
            pageCount=total_pages,
            fileCount=len(file_summaries),
            files=[UploadedFileInfo(**file_info) for file_info in file_summaries],
            message=(
                f"Processed {len(file_summaries)} file(s), "
                f"{total_pages} page(s), and {len(chunks)} chunk(s) successfully."
            ),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to process uploaded PDFs: {exc}"
        ) from exc
    finally:
        for pdf_path in staged_paths:
            if os.path.isfile(pdf_path):
                os.remove(pdf_path)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    session_id = body.sessionId.strip()
    question = body.question.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="sessionId is required.")
    if not question:
        raise HTTPException(status_code=400, detail="question must be non-empty.")
    if len(question) > 2000:
        raise HTTPException(
            status_code=400, detail="Question exceeds 2000 character limit."
        )

    store = _get_store(session_id)

    try:
        result = answer_question(store, question)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error generating answer: {exc}"
        ) from exc

    return ChatResponse(
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        isOutOfScope=result["isOutOfScope"],
    )


@app.get("/api/chat/session/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    info = _session_info.get(session_id) or _load_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found.")
    _session_info[session_id] = info
    return SessionInfo(**info)


@app.delete("/api/chat/session/{session_id}")
def delete_session(session_id: str):
    exists = FAISSStore.exists(session_id) or session_id in _session_info

    if not exists and not _load_info(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    _session_cache.pop(session_id, None)
    _session_info.pop(session_id, None)

    store = FAISSStore(session_id)
    store.delete()

    return {"message": "Session deleted."}


@app.get("/api/health")
def health():
    return {"status": "ok", "model": LLM_MODEL, "retrieval": "BM25"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=True)
