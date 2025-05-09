from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def create_hybrid_retriever(vector_store,k=4,embedding_model_name= "all-MiniLM-L6-v2"):
    """
    Create a hybrid retriever combining FAISS(semantic search) and BM25(keyword) from vector store.
    
    Args:
        vector_store: Either a FAISS vector store instance or a path to a saved vector store
        k: Number of documents to retrieve
        embedding_model_name: Name of the sentence transformer model to use for embeddings
        
    Returns:
        A hybrid retriever instance
    """
    # If vector_store is a string, load it from disk
    if isinstance(vector_store, str):
        embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        vector_store = FAISS.load_local(vector_store, embedding_function)
    
    # Create FAISS retriever
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # Extract original documents
    docs = vector_store.docstore._dict.values()

    # Create sparse retriever - BM25 Keyword search from the original documents

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k # number of documents to be retreived

    # Combine the retrievers

    hybrid_retriever = EnsembleRetriever(
                            retrievers= [faiss_retriever,bm25_retriever],
                            weights= [0.5,0.5]
                        )
    
    return hybrid_retriever
