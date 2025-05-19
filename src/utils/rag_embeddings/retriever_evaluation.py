from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from groundtruth import ground_truth
from retriever import create_hybrid_retriever
import os 
from dotenv import load_dotenv
from chunking import load_and_split_pdf
from embedding import embeddings
from vector_store import create_vector_store

load_dotenv()

API_KEY1 = os.getenv("API_KEY1")

def evaluate_rag_pipeline(retriever, ground_truth, api_key = API_KEY1,model_name= "gpt-4", k=4):
    """
    Evaluate RAG pipeline using LLM-based QA eval.
    """
    llm = ChatOpenAI(openai_api_key=api_key, model=model_name)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    predictions = []
    for item in ground_truth:
        result = qa_chain({"query": item["query"]})
        predictions.append({
            "query": item["query"],
            "reference": item["reference_answer"],
            "prediction": result["result"],
            "docs": result["source_documents"]
        })

    # Evaluate using LLM
    eval_chain = QAEvalChain.from_llm(llm)
    graded_results = eval_chain.evaluate(
        predictions,
        question_key="query",
        prediction_key="prediction",
        reference_key="reference"
    )

    # Combine scores
    scored = [
        {
            **pred,
            **grade
        } for pred, grade in zip(predictions, graded_results)
    ]

    return scored



def visualize_rag_results(results):
    df = pd.DataFrame(results)
    
    # Binary pass/fail based on LLM judgment
    df["Correct"] = df["results"].apply(lambda x: 1 if "CORRECT" in x.upper() else 0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    df["Correct"].value_counts().plot(kind="bar", ax=ax, color=["salmon", "skyblue"])
    ax.set_title("RAG Answer Correctness (LLM Judgment)")
    ax.set_xticklabels(["Incorrect", "Correct"], rotation=0)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

    return df


file_path = r"C:\Users\dceka\OneDrive\Desktop\Swathika\innovapth\ML_AI\GEN_AI\customer_care_agent_tesla\customer_care_agent\resources\Tesla_Owners_Manual_Model_X.pdf"
save_path = r"C:\Users\dceka\OneDrive\Desktop\Swathika\innovapth\ML_AI\GEN_AI\customer_care_agent_tesla\customer_care_agent\src\utils\rag_embeddings"
chunks = load_and_split_pdf(file_path)

vector_store = create_vector_store(chunks,save_path)
retriever = create_hybrid_retriever(vector_store)

rag_results = evaluate_rag_pipeline(retriever, ground_truth)
df_eval = visualize_rag_results(rag_results)
print(df_eval[["query", "prediction", "reference", "results"]])