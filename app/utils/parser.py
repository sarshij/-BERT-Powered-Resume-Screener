"""
Resume Text Parser
Extracts text from PDF and TXT files.
"""
import io

from app.logger import setup_logger

logger = setup_logger(__name__)

def parse_resume(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    logger.debug(f"Parsing: {filename} (format: {ext}, size: {len(file_bytes)} bytes)")
    if ext == 'pdf':
        return parse_pdf(file_bytes)
    elif ext == 'txt':
        return parse_txt(file_bytes)
    else:
        return parse_txt(file_bytes)

def parse_pdf(file_bytes: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype='pdf')
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        char_count = len(text.strip())
        logger.debug(f"PDF parsed: {char_count} chars")
        return text.strip()
    except ImportError:
        logger.error("PyMuPDF (fitz) not installed")
        return "[PDF parser not available]"

def parse_txt(file_bytes: bytes) -> str:
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            text = file_bytes.decode(enc).strip()
            logger.debug(f"TXT parsed: {len(text)} chars (encoding: {enc})")
            return text
        except (UnicodeDecodeError, ValueError):
            continue
    logger.warning("TXT encoding detection failed, falling back to latin-1 with replace")
    return file_bytes.decode('latin-1', errors='replace').strip()
