import os
import datetime
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from dotenv import load_dotenv

load_dotenv()

def parse_pdf(file_path: str) -> str:
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text.append(extracted)
    return "\n".join(text)


def parse_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(
        para.text for para in doc.paragraphs if para.text.strip()
    )


def parse_html(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def parse_image(file_path: str) -> str:
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)


def extract(file_path: str) -> dict:
    ext = os.path.splitext(file_path)[1].lower()

    parsers = {
        ".pdf":  parse_pdf,
        ".docx": parse_docx,
        ".html": parse_html,
        ".jpg":  parse_image,
        ".png":  parse_image,
    }

    if ext not in parsers:
        raise ValueError(f"Unsupported file format: {ext}")

    content = parsers[ext](file_path)

    return {
        "content":  content,
        "source":   os.path.basename(file_path),
        "doc_type": ext.lstrip("."),
        "date":     datetime.datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/etl_parser.py <path_to_file>")
        sys.exit(1)

    result = extract(sys.argv[1])
    print(f"Source   : {result['source']}")
    print(f"Type     : {result['doc_type']}")
    print(f"Date     : {result['date']}")
    print(f"Content  : {result['content'][:300]}...")