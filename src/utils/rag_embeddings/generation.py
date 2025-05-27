from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os 
from dotenv import load_dotenv

load_dotenv()

API_KEY1 = os.getenv("API_KEY1")  # This should be your Groq API key


def create_generator(retriever, api_key=API_KEY1, model_name="llama-3.1-8b-instant", custom_prompt=None):
    """
    Create a RAG generator chain using Groq API.
    
    Args:
        retriever: A retriever instance
        api_key: Groq API key
        model_name: Name of the LLM to use (Groq model)
        custom_prompt: Optional custom prompt template
        
    Returns:
        A RAG chain instance
    """
    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0  # You can adjust this as needed
    )
    
    # Create prompt template
    if custom_prompt:
        prompt = ChatPromptTemplate.from_template(custom_prompt)
    else:
        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant. Use the following context to answer the question.
            If you don't know the answer based on the context, say "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Question:
            {input}
            """
        )
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain