from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

class AgenticRAG:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)
        self.retriever = vectorstore.get_retriever()
        
    def get_response(self, query, history):
        # Create agents
        context_analyzer = Agent(
            role='Context Analyzer',
            goal='Analyze conversation history and expand the query appropriately',
            backstory='Expert at understanding conversation context and formulating comprehensive queries',
            llm=self.llm
        )
        
        researcher = Agent(
            role='Research Analyst',
            goal='Find relevant information from the document and track citations',
            backstory='Expert at analyzing documents and finding relevant information with proper citations. Only uses information from the provided document.',
            llm=self.llm,
            tools=[self.retriever]
        )
        
        writer = Agent(
            role='Technical Writer',
            goal='Create clear and accurate responses with citations',
            backstory='Expert at creating concise and accurate responses based strictly on research findings and maintaining proper citations',
            llm=self.llm
        )
        
        # Format history for context
        formatted_history = self._format_history(history)
        
        # Create tasks
        context_task = Task(
            description=f"""Analyze the conversation history and current query to create an expanded version that captures the full context.
            History: {formatted_history}
            Current Query: {query}
            Provide an expanded query that includes relevant context from the history.""",
            agent=context_analyzer
        )
        
        research_task = Task(
            description=f"""Research the following query and provide citations.
            Only use information from the provided document.
            If information cannot be found in the document, explicitly state that.
            Include page numbers and chunk IDs in your research notes.""",
            agent=researcher
        )
        
        writing_task = Task(
            description="""Create a comprehensive response based strictly on the research findings.
            Do not include any external knowledge.
            Include citations in a separate section.
            If the research doesn't provide sufficient information, state that clearly.""",
            agent=writer
        )
        
        # Create and run crew
        crew = Crew(
            agents=[context_analyzer, researcher, writer],
            tasks=[context_task, research_task, writing_task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Split response and citations
        parts = result.split("Sources:", 1)
        response = parts[0].strip()
        citations = "Sources:" + parts[1] if len(parts) > 1 else "No citations provided"
        
        return response, citations
    
    def _format_history(self, history):
        if not history:
            return "No previous conversation history."
        formatted = []
        for entry in history:
            formatted.append(f"Q: {entry['question']}\nA: {entry['answer']}")
        return "\n\n".join(formatted) 