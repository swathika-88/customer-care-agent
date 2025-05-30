import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# External libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Internal imports
from utils.rag_embeddings.chunking import load_and_split_pdf
from utils.rag_embeddings.vector_store import create_vector_store
from utils.rag_embeddings.retriever import create_hybrid_retriever
from utils.rag_embeddings.generation import create_generator

# Configuration
src_root = Path(__file__).parents[1]
sys.path.append(str(src_root))
load_dotenv()
API_KEY1 = os.getenv("API_KEY1")

# Global variables for configuration
DEFAULT_FAISS_PATH = "./ui/backend_fapi"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"

def validate_api_key():
    """Validate that API key is available."""
    if not API_KEY1:
        raise ValueError("Missing API key. Please check your .env file.")
    return API_KEY1

def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Get embedding model instance."""
    return HuggingFaceEmbeddings(model_name=model_name)

def process_documents(files, faiss_path: str = DEFAULT_FAISS_PATH) -> Dict[str, Any]:
    """
    Process uploaded documents and create vector store.
    
    Args:
        files: List of uploaded files
        faiss_path: Path to save FAISS index
        
    Returns:
        Dict containing processing results
    """
    validate_api_key()
    all_chunks = []
    
    try:
        # for uploaded_file in files:
        #     filename = uploaded_file.filename
            
        #     # Reset file pointer
        #     uploaded_file.file.seek(0)
            
        #     # Step 1: Chunking
        #     chunks = load_and_split_pdf(
        #         uploaded_file.file, 
        #         metadata={"source": filename}
        #     )
        #     all_chunks.extend(chunks)
        #     print('for loop process documents here -------')
        
        uploaded_file = files
        filename = uploaded_file.filename
        
        # Reset file pointer
        uploaded_file.file.seek(0)
        
        # Step 1: Chunking
        chunks = load_and_split_pdf(
            uploaded_file.file, 
            metadata={"source": filename}
        )
        all_chunks.extend(chunks)
        print('for loop process documents here -------')
        
        if not all_chunks:
            raise ValueError("No chunks were created from the uploaded files.")
        
        # Step 2: Create and save vector store
        create_and_save_vector_store(all_chunks, faiss_path)
        
        return {
            "message": "Documents processed successfully",
            # "total_chunks": len(all_chunks),
            # "total_files": len(files),
            # "filenames": [file.filename for file in files],
        }
        
    except Exception as e:
        print(f"Error processing documents: {e}")
        return {"error": str(e)}

def create_and_save_vector_store(chunks: List, faiss_path: str) -> None:
    """
    Create vector store from chunks and save to disk.
    
    Args:
        chunks: List of document chunks
        faiss_path: Path to save FAISS index
    """
    try:
        print("vector store creation is started to execute")
        # Create vector store
        vector_store = create_vector_store(documents=chunks)
        
        # Ensure directory exists
        os.makedirs(faiss_path, exist_ok=True)
        print("vector store creation is started to execute 2")
        
        # Save vector store
        vector_store.save_local(faiss_path)
        print(f"Vector store saved to {faiss_path}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to create/save vector store: {e}")

def get_retriever(faiss_path: str = DEFAULT_FAISS_PATH, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    """
    Get retriever from existing vector store.
    
    Args:
        faiss_path: Path to FAISS index
        embedding_model: Name of embedding model to use
        
    Returns:
        FAISS retriever object
    """
    embeddings = get_embeddings(embedding_model)
    index_file = os.path.join(faiss_path, "index.faiss")
    
    if os.path.exists(faiss_path) and os.path.exists(index_file):
        try:
            db = FAISS.load_local(
                faiss_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("Loaded existing FAISS index")
            return db.as_retriever()
            
        except Exception as e:
            print(f"Error loading existing index: {e}")
            raise RuntimeError(f"Failed to load FAISS index: {e}")
    else:
        raise FileNotFoundError(
            f"FAISS index not found at {faiss_path}. "
            "Please process documents first to create the index."
        )

def create_generator_from_retriever(retriever):
    """
    Create generator from retriever.
    
    Args:
        retriever: FAISS retriever object
        
    Returns:
        Generator object for question answering
    """
    try:
        return create_generator(retriever=retriever)
    except Exception as e:
        raise RuntimeError(f"Failed to create generator: {e}")

def ask_question(query: str, generator=None, faiss_path: str = DEFAULT_FAISS_PATH) -> Dict[str, Any]:
    """
    Ask a question using the RAG pipeline.
    
    Args:
        query: User question
        generator: Optional pre-created generator object
        faiss_path: Path to FAISS index (used if generator not provided)
        
    Returns:
        Dict containing answer and context
    """
    try:
        # If no generator provided, create one
        print('ask quesiton called')
        if generator is None:
            print('ask quesiton called generator not found')

            retriever = get_retriever(faiss_path)
            generator = create_generator_from_retriever(retriever)
            print('ask quesiton called generator  found and loaded')

        
        if generator is None:
            raise ValueError(
                "Generator could not be created. "
                "Please ensure documents are processed and index exists."
            )
        response = generator.invoke({"input": query})
        
        answer = response["answer"]
        print("generator invoked")
        context_docs = response.get("context", [])
        print("after context array")
        context = []
        for doc in context_docs:
            context.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        print("after for loop")
        
        return {
            "query": query,
            "answer": answer,
            "context": context
        }
        
    except Exception as e:
        print(f"Error answering question: {e}")
        return {"error": str(e)}

def evaluate_rag_pipeline(retriever, questions: List[Dict], model_name: str = DEFAULT_LLM_MODEL) -> List[Dict]:
    """
    Evaluate RAG pipeline performance.
    
    Args:
        retriever: FAISS retriever
        questions: List of evaluation questions with reference answers
        model_name: LLM model name for evaluation
        
    Returns:
        List of evaluation results
    """
    api_key = validate_api_key()
    
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name=model_name, 
        temperature=0
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True
    )
    
    results = []
    
    for item in questions:
        try:
            result = qa_chain({"query": item["query"]})
            prediction = result["result"]
            
            eval_prompt = f"""
            Question: {item['query']}
            Expected Answer: {item['reference_answers']}
            RAG Answer: {prediction}
            
            Is the RAG answer correct? Respond with only "CORRECT" or "INCORRECT".
            """
            
            evaluation = llm.predict(eval_prompt)
            is_correct = "CORRECT" in evaluation.upper()
            
            results.append({
                "query": item["query"],
                "reference": item["reference_answers"],
                "prediction": prediction,
                "is_correct": is_correct
            })
            
        except Exception as e:
            print(f"Error evaluating question '{item['query']}': {e}")
            results.append({
                "query": item["query"],
                "reference": item["reference_answers"],
                "prediction": "Error occurred",
                "is_correct": False
            })
    
    return results

def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dict containing calculated metrics
    """
    if not results:
        return {}
    
    y_true = [1] * len(results)  # All should be correct
    y_pred = [1 if r["is_correct"] else 0 for r in results]
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'correct': sum(y_pred),
        'total': len(results)
    }

def setup_rag_pipeline(files, faiss_path: str = DEFAULT_FAISS_PATH):
    """
    Complete setup function that processes documents and returns retriever and generator.
    
    Args:
        files: List of uploaded files
        faiss_path: Path to save/load FAISS index
        
    Returns:
        Tuple of (retriever, generator)
    """
    try:
        # Process documents
        result = process_documents(files, faiss_path)
        if "error" in result:
            raise RuntimeError(f"Document processing failed: {result['error']}")
        
        # Get retriever
        retriever = get_retriever(faiss_path)
        
        # Create generator
        generator = create_generator_from_retriever(retriever)
        
        print(f"RAG pipeline setup complete. Processed {result['total_chunks']} chunks from {result['total_files']} files.")
        
        return retriever, generator
        
    except Exception as e:
        print(f"Error setting up RAG pipeline: {e}")
        raise

def run_evaluation(faiss_path: str = DEFAULT_FAISS_PATH, questions: List[Dict] = None):
    """
    Run complete evaluation pipeline.
    
    Args:
        faiss_path: Path to FAISS index
        questions: List of evaluation questions
        
    Returns:
        Dict containing evaluation results and metrics
    """
    if not questions:
        print("No evaluation questions provided")
        return {}
    
    try:
        # Get retriever
        retriever = get_retriever(faiss_path)
        
        # Run evaluation
        results = evaluate_rag_pipeline(retriever, questions)
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        return {
            "results": results,
            "metrics": metrics
        }
        
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return {"error": str(e)}

