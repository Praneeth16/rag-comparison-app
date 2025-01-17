import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path
from components.pdf_processor import process_pdf
from components.rag_handler import TraditionalRAG
from components.agentic_rag_handler import AgenticRAG
from components.tracking import track_interaction
from components.guardrails import apply_guardrails
from components.vector_store import VectorStoreManager

# Load environment variables
load_dotenv()

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'document_store' not in st.session_state:
        st.session_state.document_store = None
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStoreManager()

def main():
    st.title("RAG Comparison App")
    
    initialize_session_state()
    
    # Sidebar for uploading PDF
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                st.session_state.pdf_name = uploaded_file.name
                document_store = process_pdf(uploaded_file)
                st.session_state.document_store = document_store
                st.success("PDF processed successfully!")

    # Main chat interface
    if st.session_state.document_store is None:
        st.info("Please upload a PDF document to start.")
    else:
        st.write(f"Currently analyzing: {st.session_state.pdf_name}")
        
        # Create two columns for the chat history
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Traditional RAG")
            # Display traditional RAG messages
            for message in st.session_state.messages:
                if message["type"] == "traditional":
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if "citations" in message:
                            st.info(message["citations"])

        with col2:
            st.subheader("Agentic RAG")
            # Display agentic RAG messages
            for message in st.session_state.messages:
                if message["type"] == "agentic":
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if "citations" in message:
                            st.info(message["citations"])

    # Chat input at the bottom
    if prompt := st.chat_input("Ask a question about your document"):
        if st.session_state.document_store is None:
            st.error("Please upload a PDF document first.")
        else:
            # Add user message for both columns
            st.session_state.messages.extend([
                {"role": "user", "content": prompt, "type": "traditional"},
                {"role": "user", "content": prompt, "type": "agentic"}
            ])

            # Get traditional RAG response
            track_id = track_interaction(prompt, "traditional")
            trad_rag = TraditionalRAG(st.session_state.document_store)
            
            # Get relevant history
            trad_history = st.session_state.vector_store.get_relevant_history(
                prompt, "traditional"
            )
            
            trad_response, trad_citations = trad_rag.get_response(
                prompt, 
                trad_history
            )
            validated_trad_response = apply_guardrails(trad_response)
            
            # Store conversation
            st.session_state.vector_store.add_conversation(
                prompt, validated_trad_response, "traditional"
            )
            
            # Get agentic RAG response
            agentic_rag = AgenticRAG(st.session_state.document_store)
            
            # Get relevant history
            agent_history = st.session_state.vector_store.get_relevant_history(
                prompt, "agentic"
            )
            
            agent_response, agent_citations = agentic_rag.get_response(
                prompt, 
                agent_history
            )
            validated_agent_response = apply_guardrails(agent_response)
            
            # Store conversation
            st.session_state.vector_store.add_conversation(
                prompt, validated_agent_response, "agentic"
            )
            
            # Add assistant messages for both columns
            st.session_state.messages.extend([
                {
                    "role": "assistant",
                    "content": validated_trad_response,
                    "citations": trad_citations,
                    "type": "traditional"
                },
                {
                    "role": "assistant",
                    "content": validated_agent_response,
                    "citations": agent_citations,
                    "type": "agentic"
                }
            ])
            
            # Track responses
            track_interaction(prompt, "traditional", validated_trad_response, track_id)
            track_interaction(prompt, "agentic", validated_agent_response, track_id)

if __name__ == "__main__":
    main() 