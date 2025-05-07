from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

def create_vector_store(documents, embedding_model_name= "all-MiniLM-L6-v2", save_path= None):
    """
    Create a FAISS vector store from documents.
    
    Args:
        documents: List of Document objects to store in the vector store
        embedding_model_name: Name of the sentence transformer model to use for embeddings
        save_path: Optional path to save the vector store to
        
    Returns:
        A FAISS vector store instance
    """
    # Create embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    
    # Create vector store
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_function
    )
    
    # Save vector store if path is provided
    if save_path:
        vector_store.save_local(save_path)
        print(f"Vector store saved to {save_path}")
    
    return vector_store