"""
File Upload Validation
Provides MIME type detection and file size enforcement for uploaded resumes.

FIX 2: MIME type validation — reads actual file magic bytes (not just extension).
        Whitelists only PDF, plain text, and DOCX formats.

FIX 3: File size limit — rejects files larger than MAX_FILE_BYTES (5 MB) before
        any expensive processing occurs.
"""
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

from fastapi import HTTPException

# ── Configuration ────────────────────────────────────────────────────────────
# Maximum allowed upload size: 5 MB
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB

# Whitelisted MIME types and their human-readable labels
ALLOWED_MIME_TYPES: dict[str, str] = {
    "application/pdf":                                                     "PDF",
    "text/plain":                                                          "Plain Text",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
    # Some systems report DOCX as zip (it is a ZIP archive internally)
    "application/zip":                                                     "DOCX (ZIP)",
    # Some magic libraries return this for DOCX
    "application/x-zip-compressed":                                        "DOCX (ZIP)",
}

# Extensions allowed (secondary check alongside MIME)
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}


def validate_upload(file_bytes: bytes, filename: str) -> None:
    """
    Validate an uploaded file for size and MIME type.

    Raises:
        HTTPException(413): If file exceeds MAX_FILE_BYTES.
        HTTPException(415): If file MIME type is not whitelisted.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename from the upload form.
    """
    # ── Fix 3: File size check ───────────────────────────────────────────────
    file_size = len(file_bytes)
    if file_size > MAX_FILE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB. Maximum allowed is {MAX_FILE_BYTES // (1024*1024)} MB."
        )

    # Reject completely empty files
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Fix 2: MIME type detection via file magic bytes ──────────────────────
    try:
        if MAGIC_AVAILABLE:
            detected_mime = magic.from_buffer(file_bytes[:2048], mime=True)
        else:
            detected_mime = None
    except Exception:
        # If magic detection fails for any reason, fall back to extension check only
        detected_mime = None

    # Extension-based secondary check
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if detected_mime and detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{detected_mime}'. "
                f"Allowed formats: PDF, TXT, DOCX. "
                f"Do not rename other file types to .pdf or .docx."
            )
        )

    if not detected_mime and ext not in ALLOWED_EXTENSIONS:
        # Magic failed but extension is also wrong
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension '.{ext}'. Allowed: .pdf, .txt, .docx"
        )

    # Special case: if MIME is ZIP, verify the extension is .docx
    # (genuine DOCX files are ZIPs, but a random .zip is not acceptable)
    if detected_mime in ("application/zip", "application/x-zip-compressed"):
        if ext != "docx":
            raise HTTPException(
                status_code=415,
                detail="ZIP files are not accepted. Please upload a valid PDF, TXT, or DOCX file."
            )
