from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from components.vector_store import VectorStoreManager

class DocumentChunk:
    def __init__(self, text, page_number, chunk_id):
        self.text = text
        self.page_number = page_number
        self.chunk_id = chunk_id
        self.metadata = {
            "page_number": page_number,
            "chunk_id": chunk_id
        }

def process_pdf(uploaded_file):
    # Read PDF content
    pdf_reader = PdfReader(uploaded_file)
    documents = []
    
    for page_num, page in enumerate(pdf_reader.pages, 1):
        text = page.extract_text()
        documents.append({
            "text": text,
            "page_number": page_num
        })
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc["text"])
        for chunk in doc_chunks:
            chunk_id = str(uuid.uuid4())
            chunks.append(
                DocumentChunk(
                    text=chunk,
                    page_number=doc["page_number"],
                    chunk_id=chunk_id
                )
            )
    
    # Add chunks to vector store
    vector_store = VectorStoreManager()
    return vector_store.add_pdf_chunks(chunks, uploaded_file.name) 