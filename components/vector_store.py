from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import uuid
from datetime import datetime

class VectorStoreManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance.embeddings = OpenAIEmbeddings()
            cls._instance.store = Chroma(
                collection_name="unified_store",
                embedding_function=cls._instance.embeddings,
                persist_directory="./vector_store_db"
            )
        return cls._instance
    
    def add_pdf_chunks(self, chunks, file_name):
        texts = [chunk.text for chunk in chunks]
        metadatas = [{
            **chunk.metadata,
            "type": "pdf_chunk",
            "file_name": file_name
        } for chunk in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        self.store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        self.store.persist()
        return self.store
    
    def add_conversation(self, question, answer, rag_type):
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        conversation_text = f"Question: {question}\nAnswer: {answer}"
        
        self.store.add_texts(
            texts=[conversation_text],
            metadatas=[{
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "rag_type": rag_type,
                "type": "conversation",
                "question": question,
                "answer": answer
            }],
            ids=[conversation_id]
        )
        self.store.persist()
    
    def get_relevant_history(self, query, rag_type, k=5):
        results = self.store.similarity_search_with_metadata(
            query=query,
            k=k,
            filter={
                "type": "conversation",
                "rag_type": rag_type
            }
        )
        
        history = []
        for doc in results:
            history.append({
                "question": doc.metadata["question"],
                "answer": doc.metadata["answer"],
                "timestamp": doc.metadata["timestamp"]
            })
        
        history.sort(key=lambda x: x["timestamp"])
        return history
    
    def get_retriever(self, file_name=None):
        # Create a retriever that only searches PDF chunks
        search_kwargs = {"k": 3}
        filter_dict = {"type": "pdf_chunk"}
        
        if file_name:
            filter_dict["file_name"] = file_name
            
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
            filter=filter_dict
        ) 