from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class TraditionalRAG:
    def __init__(self, vectorstore):
        self.llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)
        
        # Custom prompt template that includes citation requirement and context control
        template = """
        Answer the following question based ONLY on the provided context. 
        If the answer cannot be fully derived from the context, say "I cannot answer this question based on the provided document."
        Include specific citations from the source material in your response.
        
        Previous conversation history for context:
        {chat_history}
        
        Current context: {context}
        
        Current question: {question}
        
        Please provide a detailed answer with citations referencing the specific chunks and page numbers used.
        Remember to only use information from the provided context and not any external knowledge.
        """
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question", "chat_history"]
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.get_retriever(),
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            memory=self.memory,
            return_source_documents=True,
            return_generated_question=True
        )
        
    def get_response(self, query, history):
        # Format history for context
        formatted_history = self._format_history(history)
        
        # Generate expanded query based on history
        expanded_query = self._expand_query(query, formatted_history)
        
        result = self.chain.invoke({
            "question": expanded_query,
        })
        
        # Format citations
        citations = []
        for doc in result["source_documents"]:
            citations.append(
                f"Page {doc.metadata['page_number']}, "
                f"Chunk ID: {doc.metadata['chunk_id']}"
            )
        
        citations_text = "Sources:\n" + "\n".join(citations)
        
        return result["answer"], citations_text
    
    def _format_history(self, history):
        if not history:
            return ""
        formatted = []
        for entry in history:
            formatted.append(f"Q: {entry['question']}\nA: {entry['answer']}")
        return "\n\n".join(formatted)
    
    def _expand_query(self, query, history):
        if not history:
            return query
            
        expansion_prompt = f"""
        Given the conversation history and the current question, create an expanded version of the question that includes relevant context from the history.
        
        History:
        {history}
        
        Current Question: {query}
        
        Expanded Question:"""
        
        expanded = self.llm.invoke(expansion_prompt)
        return expanded 