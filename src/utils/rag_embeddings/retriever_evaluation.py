from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
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

def evaluate_rag_pipeline_simple(retriever, ground_truth, api_key=API_KEY1, model_name="gpt-4"):
    """
    Evaluate RAG pipeline using a simpler direct LLM evaluation approach.
    """
    # Create the chat model for both QA and evaluation
    llm = ChatOpenAI(openai_api_key=api_key, model=model_name)
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # This will store our evaluation results
    results = []
    
    # Process each ground truth item
    for item in ground_truth:
        query = item["query"]
        reference_answer = item["reference_answers"]
        
        # Get the prediction from our RAG pipeline
        result = qa_chain({"query": query})
        prediction = result["result"]
        source_docs = result["source_documents"]
        
        # Create an evaluation prompt
        eval_prompt = f"""
        Please evaluate if the following answer correctly addresses the question based on the reference answer.
        
        Question: {query}
        
        Reference Answer: {reference_answer}
        
        Predicted Answer: {prediction}
        
        Determine if the predicted answer is CORRECT or INCORRECT based on the reference.
        Explain your reasoning briefly, then end with either "CORRECT" or "INCORRECT".
        """
        
        # Get the evaluation from the LLM
        evaluation = llm.predict(eval_prompt)
        
        # Determine if the answer was correct based on the evaluation
        is_correct = "CORRECT" in evaluation.upper()
        
        # Store the results
        results.append({
            "query": query,
            "reference": reference_answer,
            "prediction": prediction,
            "docs": source_docs,
            "results": evaluation,
            "is_correct": is_correct
        })
    
    return results


def visualize_rag_results(results):
    df = pd.DataFrame(results)
    
    # Binary pass/fail based on LLM judgment
    df["Correct"] = df["is_correct"].astype(int)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create a new DataFrame with both categories guaranteed
    plot_data = pd.DataFrame({
        'Category': ['Incorrect', 'Correct'],
        'Count': [
            sum(df["Correct"] == 0),  # Count of incorrect
            sum(df["Correct"] == 1)   # Count of correct
        ]
    })
    
    # Plot the bars with fixed colors
    ax.bar(plot_data['Category'], plot_data['Count'], color=["salmon", "skyblue"])
    
    ax.set_title("RAG Answer Correctness (LLM Judgment)")
    ax.set_ylabel("Count")
    
    # Add percentage labels on the bars
    total = len(df)
    for i, row in enumerate(plot_data.itertuples()):
        count = row.Count
        percentage = 100 * count / total if total > 0 else 0
        ax.text(i, count + 0.1, f"{percentage:.1f}%", ha="center")
    
    plt.tight_layout()
    plt.show()

    return df

