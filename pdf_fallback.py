"""
pdf_fallback.py - PDF -> Claude content blocks, with scanned-page fallback.

Install:  pip install pymupdf pdfplumber
(PyMuPDF renders pages without Poppler, so it works on Windows out of the box.)

Idea: go page by page. Pages with enough extractable text are sent as text;
pages with little/no text (scanned, or image-only slides) are rendered to JPEG
and sent as images. Mixed PDFs therefore work too, and page order is preserved.
"""
import base64
import logging

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3
except ImportError:  # older versions
    import fitz
import pdfplumber

log = logging.getLogger(__name__)

MIN_CHARS_PER_PAGE = 40   # below this, treat the page as scanned/image-only
MAX_PAGES = 30            # safety cap on cost/latency per PDF
RENDER_DPI = 130          # readable for slides without huge token cost
MAX_LONG_EDGE = 1568      # Claude downsizes beyond this anyway


def _render_page_jpeg_b64(doc: "fitz.Document", page_index: int) -> str:
    page = doc[page_index]
    zoom = RENDER_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    # downscale if the long edge is too big
    long_edge = max(pix.width, pix.height)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom * scale, zoom * scale), alpha=False)
    return base64.standard_b64encode(pix.tobytes("jpeg", jpg_quality=80)).decode()


def pdf_to_content_blocks(pdf_path: str) -> tuple[list[dict], dict]:
    """
    Returns (content_blocks, stats).
    content_blocks: list of Anthropic message content blocks (text + image),
                    ready to extend into the user message.
    stats: {"pages": n, "text_pages": n, "image_pages": n, "truncated": bool}
    """
    blocks: list[dict] = []
    text_pages = image_pages = 0

    with pdfplumber.open(pdf_path) as plumber_pdf, fitz.open(pdf_path) as mu_doc:
        total = len(plumber_pdf.pages)
        n = min(total, MAX_PAGES)

        for i in range(n):
            text = (plumber_pdf.pages[i].extract_text() or "").strip()
            if len(text) >= MIN_CHARS_PER_PAGE:
                blocks.append({"type": "text", "text": f"--- Slide/Page {i + 1} ---\n{text}"})
                text_pages += 1
            else:
                blocks.append({"type": "text", "text": f"--- Slide/Page {i + 1} (image) ---"})
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _render_page_jpeg_b64(mu_doc, i),
                    },
                })
                image_pages += 1

    stats = {
        "pages": total,
        "text_pages": text_pages,
        "image_pages": image_pages,
        "truncated": total > MAX_PAGES,
    }
    log.info("PDF %s: %s", pdf_path, stats)
    return blocks, stats
