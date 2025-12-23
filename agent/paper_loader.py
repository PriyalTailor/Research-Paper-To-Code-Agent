import os
import requests
from pypdf import PdfReader
from urllib.parse import urlparse

def is_url(input_path: str) -> bool:
    return input_path.startswith("http://") or input_path.startswith("https://")

def download_pdf_from_url(url: str, output_dir="generated") -> str:
    """
    Download PDF from an open-access paper link (e.g., arXiv).
    Returns local file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Handle arXiv abstract → PDF
    if "arxiv.org/abs/" in url:
        url = url.replace("/abs/", "/pdf/") + ".pdf"

    response = requests.get(url)
    response.raise_for_status()

    pdf_path = os.path.join(output_dir, "downloaded_paper.pdf")
    with open(pdf_path, "wb") as f:
        f.write(response.content)

    return pdf_path

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

def load_paper(paper_input: str) -> str:
    """
    Unified paper loader: supports local PDF or open paper URL.
    """
    if is_url(paper_input):
        print("🌐 Detected URL. Downloading paper...")
        pdf_path = download_pdf_from_url(paper_input)
    else:
        pdf_path = paper_input

    return load_pdf(pdf_path)
