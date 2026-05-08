"""
Split extracted PDF pages into token-bounded chunks.

Strategy
--------
- Encoding: cl100k_base (same tokeniser family as Claude / GPT-4).
- Each page is tokenised independently so chunk page_num metadata is exact.
- Long pages are split with a sliding window of CHUNK_MAX_TOKENS tokens and
  CHUNK_OVERLAP_TOKENS overlap so context is not lost at boundaries.
- Short pages that fall below CHUNK_MIN_TOKENS are still kept as single chunks.

Output
------
List of dicts:
    {
        "file_name":   str,   # original PDF file name
        "page_num":    int,   # 1-based page number from the PDF
        "text":        str,   # decoded chunk text
        "token_count": int,   # number of tokens in this chunk
    }
"""

from __future__ import annotations

import logging
import re

import tiktoken

from config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_MIN_TOKENS

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+\s*|\s+")


class _OfflineEncoding:
    """Approximate tokeniser used when tiktoken assets are unavailable."""

    def encode(self, text: str) -> list[str]:
        return _WORD_RE.findall(text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


_enc: tiktoken.Encoding | _OfflineEncoding | None = None


def _get_encoding() -> tiktoken.Encoding | _OfflineEncoding:
    global _enc

    if _enc is not None:
        return _enc

    try:
        _enc = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        logger.warning(
            "Falling back to offline approximate chunking because "
            "cl100k_base could not be loaded: %s",
            exc,
        )
        _enc = _OfflineEncoding()

    return _enc


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Convert a list of per-page text dicts into token-bounded chunks.

    Args:
        pages: Output of pdf_processor.extract_pages().

    Returns:
        List of chunk dicts with file_name, page_num, text, and token_count.
    """
    chunks: list[dict] = []

    for page in pages:
        page_chunks = _chunk_text(
            text=page["text"],
            page_num=page["page_num"],
            file_name=page.get("file_name", "document.pdf"),
        )
        chunks.extend(page_chunks)

    return chunks


def _chunk_text(text: str, page_num: int, file_name: str) -> list[dict]:
    enc = _get_encoding()
    tokens = enc.encode(text)

    if not tokens:
        return []

    # Entire page fits in one chunk
    if len(tokens) <= CHUNK_MAX_TOKENS:
        return [
            {
                "file_name": file_name,
                "page_num": page_num,
                "text": text,
                "token_count": len(tokens),
            }
        ]

    # Sliding-window split
    stride = CHUNK_MAX_TOKENS - CHUNK_OVERLAP_TOKENS
    result: list[dict] = []

    for start in range(0, len(tokens), stride):
        window = tokens[start : start + CHUNK_MAX_TOKENS]
        if len(window) < CHUNK_MIN_TOKENS:
            # Tail is too small; merge into previous chunk if one exists
            if result:
                prev = result[-1]
                merged_tokens = enc.encode(prev["text"]) + window
                prev["text"] = enc.decode(merged_tokens)
                prev["token_count"] = len(merged_tokens)
            break

        result.append(
            {
                "file_name": file_name,
                "page_num": page_num,
                "text": enc.decode(window),
                "token_count": len(window),
            }
        )

    return result
