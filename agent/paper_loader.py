from pypdf import PdfReader

def load_pdf(pdf_path: str) -> str:
    """
    Load a research paper PDF and extract raw text.
    """
    reader = PdfReader(pdf_path)
    text = []

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text.append(content)

    return "\n".join(text)
