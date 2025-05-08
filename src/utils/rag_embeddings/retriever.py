from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import create_vector_store
from chunking import load_and_split_pdf

def create_hybrid_retriever(vector_store,k=4,embedding_model_name= "all-MiniLM-L6-v2"):
    """
    Create a hybrid retriever combining FAISS(semantic search) and BM25(keyword) from vector store.
    
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

file_path = "C:/Users/dceka/OneDrive/Desktop/Swathika/innovapth/ML_AI/GEN_AI/temp/customer_care_agent/Tesla.pdf"
chunks = load_and_split_pdf(file_path)
vector_store = create_vector_store(chunks)
hybrid_retriever = create_hybrid_retriever(vector_store=vector_store)
results = hybrid_retriever.get_relevant_documents("How fast can I charge my Tesla model Y?")
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content}")