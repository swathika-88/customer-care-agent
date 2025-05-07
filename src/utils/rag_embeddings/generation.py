from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os 
from dotenv import load_dotenv


load_dotenv()

API_KEY1 = os.getenv("API_KEY1")


def create_generator(retriever,api_key = API_KEY1,model_name= "gpt-4",custom_prompt= None):
    """
    Create a RAG generator chain.
    
    Args:
        retriever: A retriever instance
        api_key: OpenAI API key
        model_name: Name of the LLM to use
        custom_prompt: Optional custom prompt template
        
    Returns:
        A RAG chain instance
    """
    # Initialize LLM
    llm = ChatOpenAI(openai_api_key=api_key, model=model_name)
    
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