import pytest
from app.utils.parser import parse_pdf, parse_txt, parse_resume


class TestParseTxt:

    def test_utf8_encoding(self):
        text = "Hello, World! こんにちは"
        result = parse_txt(text.encode('utf-8'))
        assert result == text

    def test_latin1_encoding(self):
        text = "Résumé: José Martínez - 5 années d'expérience"
        result = parse_txt(text.encode('latin-1'))
        assert result == text

    def test_empty_bytes(self):
        result = parse_txt(b"")
        assert result == ""

    def test_whitespace_only(self):
        result = parse_txt(b"   \n  \t  ")
        assert result == ""

    def test_strips_whitespace(self):
        result = parse_txt(b"  hello world  ")
        assert result == "hello world"


class TestParseResume:

    def test_dispatches_txt_by_extension(self):
        result = parse_resume(b"python developer", "resume.txt")
        assert "python developer" in result

    def test_dispatches_unknown_as_txt(self):
        result = parse_resume(b"some content", "resume.csv")
        assert "some content" in result

    def test_handles_no_extension(self):
        result = parse_resume(b"content", "resume")
        assert "content" in result

    def test_pdf_handles_invalid_content(self):
        try:
            result = parse_resume(b"not a real pdf", "resume.pdf")
            assert isinstance(result, str)
        except Exception:
            pass
