import io
import uuid
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentIngestor:
    def __init__(self):
        # We use LangChain's RecursiveCharacterTextSplitter for optimal NLP chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    def parse_pdf(self, file_bytes: bytes) -> str:
        """Extract text from a PDF file."""
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n\n"
        return text

    def parse_txt(self, file_bytes: bytes) -> str:
        """Extract text from a TXT file."""
        return file_bytes.decode('utf-8')

    def process_document(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """Parse a document and return a list of chunks ready for vector DB upsertion."""
        # 1. Parse Text
        if filename.lower().endswith('.pdf'):
            text = self.parse_pdf(file_bytes)
        elif filename.lower().endswith('.txt'):
            text = self.parse_txt(file_bytes)
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")
            
        if not text.strip():
            raise ValueError("No text could be extracted from the document.")

        # 2. Chunk Text
        chunks = self.text_splitter.split_text(text)
        
        # 3. Format for VectorDB
        source_id = f"doc_{uuid.uuid4().hex[:8]}"
        vdb_chunks = []
        for i, chunk_text in enumerate(chunks):
            vdb_chunks.append({
                "id": f"{source_id}_chunk_{i}",
                "text": chunk_text,
                "strategy": "user_uploaded_document",
                "metadata": {
                    "source_id": source_id,
                    "filename": filename,
                    "passage_index": i
                }
            })
            
        return vdb_chunks
