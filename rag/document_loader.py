import fitz
from docx import Document
import io


def load_document(file_bytes, filename):
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        return text

    elif ext == "docx":
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    elif ext == "txt":
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {ext}")