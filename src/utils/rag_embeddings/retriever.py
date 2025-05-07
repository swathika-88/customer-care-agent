from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

def create_retriever(vector_store,k=4,embedding_model_name= "all-MiniLM-L6-v2"):
    """
    Create a retriever from a vector store.
    
    Args:
        vector_store: Either a FAISS vector store instance or a path to a saved vector store
        k: Number of documents to retrieve
        embedding_model_name: Name of the sentence transformer model to use for embeddings
        
    Returns:
        A retriever instance
    """
    # If vector_store is a string, load it from disk
    if isinstance(vector_store, str):
        embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        vector_store = FAISS.load_local(vector_store, embedding_function)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_type = "similarity",search_kwargs={"k": k})
    
    return retriever
