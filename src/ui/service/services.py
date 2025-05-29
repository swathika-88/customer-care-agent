import os
from dotenv import load_dotenv
from utils.rag_embeddings.chunking import load_and_split_pdf
from utils.rag_embeddings.vector_store import create_vector_store
from utils.rag_embeddings.retriever import create_hybrid_retriever
from utils.rag_embeddings.generation import create_generator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import sys
from pathlib import Path

src_root = Path(__file__).parents[1]  # Go up 1 level to reach src
sys.path.append(str(src_root))
load_dotenv()
API_KEY1 = os.getenv("API_KEY1")

def process_documents(files):
    if not API_KEY1:
        raise ValueError("Missing API key")

    all_chunks = []
    for uploaded_file in files:
        filename = uploaded_file.filename
        content = uploaded_file.file.read()

        # Create a dummy file-like object
        file_like = uploaded_file.file
        file_like.seek(0)  # ensure pointer is at beginning

        # Step 1: Chunking
        chunks = load_and_split_pdf(file_like, metadata={"source": filename})
        all_chunks.extend(chunks)

    # Step 2: Create vector store
    vector_store = create_vector_store(documents=all_chunks)

    # Step 3: Create retriever and generator
    retriever = create_hybrid_retriever(vector_store=vector_store)
    generator = create_generator(retriever=retriever)

    return {
        "message": "Documents processed successfully",
        "total_chunks": len(all_chunks),
        "total_files": len(files),
        "filenames": [file.filename for file in files],
    }

# services/chatbot_service.py

def ask_question(generator, query: str):
    if generator is None:
        raise ValueError("Generator is not initialized. Upload and process documents first.")

    response = generator.invoke({"input": query})

    answer = response["answer"]
    context_docs = response.get("context", [])
    context = []

    for doc in context_docs:
        context.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    return {
        "query": query,
        "answer": answer,
        "context": context
    }

def evaluate_rag_pipeline(retriever, questions, model_name="llama-3.1-8b-instant", api_key=API_KEY1):
    llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    results = []
    for item in questions:
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

    return results

def calculate_metrics(results):
    if not results:
        return {}
    
    y_true = [1] * len(results)
    y_pred = [1 if r["is_correct"] else 0 for r in results]

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'correct': sum(y_pred),
        'total': len(results)
    }


def get_retriever():
    # Path to your FAISS index
    faiss_path = "customer_care_agent/src/ui"

    # Initialize embedding model (you can use any other as needed)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load vector store
    if not os.path.exists(faiss_path):
        raise ValueError(f"Vector store path '{faiss_path}' does not exist.")
    
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    
    return db.as_retriever()
