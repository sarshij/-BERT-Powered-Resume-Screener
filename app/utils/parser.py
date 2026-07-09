"""
Resume Text Parser
Extracts text from PDF, TXT and DOCX files.
Features Layout-aware parsing, OCR fallback, and Language Detection.
"""
import io
from app.logger import setup_logger
from langdetect import detect, LangDetectException

logger = setup_logger(__name__)

def parse_resume(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    logger.debug(f"Parsing: {filename} (format: {ext}, size: {len(file_bytes)} bytes)")
    
    text = ""
    if ext == 'pdf':
        text = parse_pdf(file_bytes)
    elif ext == 'txt':
        text = parse_txt(file_bytes)
    elif ext in ['docx', 'doc']:
        text = parse_docx(file_bytes)
    else:
        text = parse_txt(file_bytes)
        
    text = text.strip()
    
    # Language Detection
    if text:
        try:
            lang = detect(text)
            logger.debug(f"Detected language for {filename}: {lang}")
            # Multilingual SBERT will handle different languages natively
        except LangDetectException:
            logger.warning(f"Could not detect language for {filename}")
            
    return text

def parse_pdf(file_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed")
        return "[PDF parser not available]"
        
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                # Extract text preserving layout/columns
                page_text = page.extract_text(layout=True)
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"pdfplumber failed: {e}")
        
    text = text.strip()
    
    # OCR Fallback for scanned PDFs
    if len(text) < 50:
        logger.info("PDF text too short, attempting OCR fallback...")
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            
            images = convert_from_bytes(file_bytes)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img) + "\n"
                
            if len(ocr_text.strip()) > len(text):
                logger.info("OCR fallback successful")
                return ocr_text.strip()
        except ImportError:
            logger.error("pytesseract or pdf2image not installed for OCR fallback")
        except Exception as e:
            logger.error(f"OCR fallback failed: {e}")
            
    return text

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

def parse_docx(file_bytes: bytes) -> str:
    try:
        import mammoth
    except ImportError:
        logger.error("mammoth not installed")
        return "[DOCX parser not available]"
        
    try:
        # Mammoth converts docx to plain text while preserving structure better
        result = mammoth.extract_raw_text(io.BytesIO(file_bytes))
        text = result.value
        logger.debug(f"DOCX parsed (mammoth): {len(text.strip())} chars")
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to parse DOCX with mammoth: {e}")
        # Fallback to python-docx if mammoth fails
        try:
            import docx
            document = docx.Document(io.BytesIO(file_bytes))
            text = '\n'.join([paragraph.text for paragraph in document.paragraphs])
            logger.debug(f"DOCX parsed (fallback): {len(text.strip())} chars")
            return text.strip()
        except Exception as fallback_e:
            logger.error(f"Fallback python-docx also failed: {fallback_e}")
            return "[DOCX parser error]"
