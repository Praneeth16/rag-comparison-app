# RAG Comparison App

A Streamlit application that compares Traditional RAG (Retrieval-Augmented Generation) with Agentic RAG using CrewAI. This application allows users to upload PDF documents and ask questions about them while comparing the responses from both approaches.

## Features

- PDF document processing and chunking
- Dual RAG implementation:
  - Traditional RAG using LangChain
  - Agentic RAG using CrewAI
- Semantic conversation history tracking
- Unified vector store for both PDF chunks and conversation history
- Response quality control using Guardrails
- Interaction tracking using Opik
- Side-by-side comparison of responses
- Citation tracking and display

## Installation

1. Clone the repository:
```git clone https://github.com/Praneeth16/rag-comparison-app.git)```
```cd rag-comparison-app```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create a `.env` file in the project root and add your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key
OPIK_API_KEY=your_opik_api_key
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Upload a PDF document using the sidebar

3. Ask questions about the document using the chat interface

4. Compare responses from both RAG implementations side by side

## Project Structure

```
rag-comparison-app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Project dependencies
├── .env                       # Environment variables
├── components/
│   ├── __init__.py
│   ├── pdf_processor.py       # PDF processing utilities
│   ├── vector_store.py        # Unified vector store manager
│   ├── rag_handler.py         # Traditional RAG implementation
│   ├── agentic_rag_handler.py # Agentic RAG implementation
│   ├── tracking.py            # Interaction tracking
│   └── guardrails.py          # Response quality control
└── vector_store_db/           # Vector store database directory
```

## Features in Detail

### Traditional RAG
- Uses LangChain's ConversationalRetrievalChain
- Implements semantic search for relevant document chunks
- Maintains conversation history for context
- Provides source citations

### Agentic RAG
- Uses CrewAI for agent-based interactions
- Implements three specialized agents:
  - Context Analyzer: Understands conversation history
  - Research Analyst: Finds relevant information
  - Technical Writer: Creates comprehensive responses
- Provides detailed citations and explanations

### Vector Store
- Unified ChromaDB instance for both PDF chunks and conversations
- Metadata-based filtering for different types of content
- Semantic search capabilities
- Persistent storage

### Quality Control
- Guardrails implementation to prevent hallucinations
- Strict context adherence
- Citation requirements
- Response validation
