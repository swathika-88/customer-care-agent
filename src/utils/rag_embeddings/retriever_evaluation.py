from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
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
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import seaborn as sns

load_dotenv()

API_KEY1 = os.getenv("API_KEY1")

def evaluate_rag_pipeline_simple(retriever, ground_truth, api_key=API_KEY1, model_name="llama-3.1-8b-instant"):
    """
    Evaluate RAG pipeline using a simpler direct LLM evaluation approach with Groq API.
    """
    # Create the chat model for both QA and evaluation
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0
    )
    
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


def calculate_metrics(results):
    """
    Calculate precision, recall, accuracy, and F1 score from evaluation results.
    """
    # Convert boolean results to binary (1 for correct, 0 for incorrect)
    y_true = [1] * len(results)  # All ground truth answers are correct by definition
    y_pred = [1 if result["is_correct"] else 0 for result in results]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'total_questions': len(results),
        'correct_answers': sum(y_pred),
        'incorrect_answers': len(results) - sum(y_pred)
    }
    
    return metrics


def visualize_rag_results(results):
    """
    Visualize RAG results with metrics and confusion matrix.
    """
    df = pd.DataFrame(results)
    
    # Binary pass/fail based on LLM judgment
    df["Correct"] = df["is_correct"].astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Answer Correctness Bar Chart
    plot_data = pd.DataFrame({
        'Category': ['Incorrect', 'Correct'],
        'Count': [
            metrics['incorrect_answers'],
            metrics['correct_answers']
        ]
    })
    
    ax1.bar(plot_data['Category'], plot_data['Count'], color=["salmon", "skyblue"])
    ax1.set_title("RAG Answer Correctness (LLM Judgment)")
    ax1.set_ylabel("Count")
    
    # Add percentage labels on the bars
    total = len(df)
    for i, row in enumerate(plot_data.itertuples()):
        count = row.Count
        percentage = 100 * count / total if total > 0 else 0
        ax1.text(i, count + 0.1, f"{percentage:.1f}%", ha="center")
    
    # Plot 2: Metrics Bar Chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
    
    bars = ax2.bar(metric_names, metric_values, color=['lightgreen', 'lightblue', 'lightcoral', 'lightyellow'])
    ax2.set_title("Performance Metrics")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 3: Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Predicted Incorrect', 'Predicted Correct'],
                yticklabels=['Actually Correct', 'Actually Correct'])
    ax3.set_title("Confusion Matrix")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    
    # Plot 4: Metrics Summary Text
    ax4.axis('off')
    summary_text = f"""
    Performance Summary:
    
    Total Questions: {metrics['total_questions']}
    Correct Answers: {metrics['correct_answers']}
    Incorrect Answers: {metrics['incorrect_answers']}
    
    Accuracy: {metrics['accuracy']:.3f}
    Precision: {metrics['precision']:.3f}
    Recall: {metrics['recall']:.3f}
    F1 Score: {metrics['f1_score']:.3f}
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed metrics
    print("="*50)
    print("DETAILED EVALUATION METRICS")
    print("="*50)
    print(f"Total Questions Evaluated: {metrics['total_questions']}")
    print(f"Correct Answers: {metrics['correct_answers']}")
    print(f"Incorrect Answers: {metrics['incorrect_answers']}")
    print("-"*30)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print("="*50)
    
    return df, metrics


def detailed_results_analysis(results):
    """
    Provide detailed analysis of individual results.
    """
    print("\n" + "="*80)
    print("DETAILED RESULTS ANALYSIS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        status = "✅ CORRECT" if result["is_correct"] else "❌ INCORRECT"
        print(f"\n[Question {i}] {status}")
        print(f"Query: {result['query']}")
        print(f"Reference: {result['reference']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Evaluation: {result['results']}")
        print("-" * 80)


# Example usage function
def run_full_evaluation(pdf_path, ground_truth_data):
    """
    Run complete evaluation pipeline.
    """
    # Load and process documents
    chunks = load_and_split_pdf(pdf_path)
    vector_store = create_vector_store(chunks, embeddings)
    retriever = create_hybrid_retriever(vector_store, chunks)
    
    # Run evaluation
    results = evaluate_rag_pipeline_simple(retriever, ground_truth_data)
    
    # Visualize and analyze results
    df, metrics = visualize_rag_results(results)
    
    # Show detailed analysis
    detailed_results_analysis(results)
    
    return results, metrics