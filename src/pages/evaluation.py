import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json
import tempfile
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import numpy as np

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY1")

# Import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from pages import upload_file, chatbot

def evaluate_rag_pipeline(retriever, ground_truth, api_key=API_KEY, model_name="llama-3.1-8b-instant"):
    """
    Evaluate RAG pipeline using Groq API with comprehensive metrics.
    """
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

    for item in ground_truth:
        query = item["query"]
        reference_answer = item["reference_answers"]
        result = qa_chain({"query": query})
        prediction = result["result"]
        source_docs = result["source_documents"]

        eval_prompt = f"""
        Please evaluate if the following answer correctly addresses the question based on the reference answer.
        
        Question: {query}
        
        Reference Answer: {reference_answer}
        
        Predicted Answer: {prediction}
        
        Determine if the predicted answer is CORRECT or INCORRECT based on the reference.
        Explain your reasoning briefly, then end with either "CORRECT" or "INCORRECT".
        """
        
        evaluation = llm.predict(eval_prompt)
        is_correct = "CORRECT" in evaluation.upper()

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
    Calculate comprehensive evaluation metrics.
    """
    if not results:
        return None
    
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
    Create comprehensive visualization of RAG evaluation results.
    """
    if not results:
        return None, None

    df = pd.DataFrame(results)
    df["Correct"] = df["is_correct"].astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Answer Correctness Bar Chart
    plot_data = pd.DataFrame({
        'Category': ['Incorrect', 'Correct'],
        'Count': [
            metrics['incorrect_answers'],
            metrics['correct_answers']
        ]
    })
    
    ax1.bar(plot_data['Category'], plot_data['Count'], color=["salmon", "skyblue"])
    ax1.set_title("RAG Answer Correctness")
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
    summary_text = f"""Performance Summary:

Total Questions: {metrics['total_questions']}
Correct Answers: {metrics['correct_answers']}
Incorrect Answers: {metrics['incorrect_answers']}

Accuracy: {metrics['accuracy']:.3f}
Precision: {metrics['precision']:.3f}
Recall: {metrics['recall']:.3f}
F1 Score: {metrics['f1_score']:.3f}"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig, df, metrics

def render():
    st.header("🔍 RAG System Evaluation")
    st.markdown("Evaluate your RAG system's performance with comprehensive metrics using Groq API")

    if "retriever" not in st.session_state or not st.session_state.retriever:
        st.warning("⚠️ Please upload and process PDF documents in the 'File Upload' tab first.")
        return

    if not API_KEY:
        st.error("❌ Groq API key not found in environment. Please check your .env file.")
        return

    # Model selection
    model_options = [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile", 
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ]
    selected_model = st.selectbox("Select Groq Model:", model_options)

    st.markdown("---")
    
    # Load existing ground truth
    try:
        from utils.rag_embeddings.groundtruth import ground_truth
        st.success(f"✅ Loaded {len(ground_truth)} ground truth items from file.")

        with st.expander("📋 Preview Ground Truth Data"):
            for i, item in enumerate(ground_truth[:3]):  # Show first 3 items
                st.write(f"**Q{i+1}:** {item['query']}")
                st.write(f"**Reference Answer:** {item['reference_answers']}")
                st.write("---")
            if len(ground_truth) > 3:
                st.write(f"... and {len(ground_truth) - 3} more items")

    except Exception as e:
        st.warning(f"⚠️ Could not load ground truth file: {e}")
        ground_truth = []

    # Manual ground truth creation
    st.subheader("📝 Add Custom QA Pairs")
    
    if "manual_gt" not in st.session_state:
        st.session_state.manual_gt = []

    with st.form("add_qa_pair"):
        col1, col2 = st.columns(2)
        with col1:
            query = st.text_area("Question", height=100, placeholder="Enter your question here...")
        with col2:
            reference = st.text_area("Reference Answer", height=100, placeholder="Enter the expected answer...")
        
        if st.form_submit_button("➕ Add QA Pair") and query and reference:
            st.session_state.manual_gt.append({
                "query": query,
                "reference_answers": reference
            })
            st.success("✅ QA pair added!")
            st.rerun()

    # Display current ground truth items
    if st.session_state.manual_gt:
        st.subheader("📚 Current Ground Truth Items")
        for i, item in enumerate(st.session_state.manual_gt):
            with st.expander(f"QA Pair {i+1}: {item['query'][:50]}..."):
                st.write(f"**Question:** {item['query']}")
                st.write(f"**Reference:** {item['reference_answers']}")
                if st.button(f"🗑️ Remove Pair #{i+1}", key=f"remove_{i}"):
                    st.session_state.manual_gt.pop(i)
                    st.rerun()

        if st.button("💾 Save Ground Truth to File"):
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump(st.session_state.manual_gt, f, indent=2)
            with open(f.name, 'r') as f:
                st.download_button(
                    "📥 Download JSON", 
                    data=f.read(), 
                    file_name="ground_truth.json", 
                    mime="application/json"
                )

        # Use manual ground truth if available
        if st.session_state.manual_gt:
            ground_truth = st.session_state.manual_gt

    st.markdown("---")
    
    # Run evaluation
    if ground_truth and len(ground_truth) > 0:
        st.subheader("🚀 Run Evaluation")
        st.info(f"Ready to evaluate {len(ground_truth)} questions using {selected_model}")
        
        if st.button("▶️ Start Evaluation", type="primary"):
            with st.spinner("🔄 Evaluating RAG system... This may take a few minutes."):
                try:
                    eval_results = evaluate_rag_pipeline(
                        st.session_state.retriever, 
                        ground_truth,
                        model_name=selected_model
                    )
                    st.session_state.eval_results = eval_results
                    st.session_state.selected_model = selected_model
                    st.success("✅ Evaluation complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
    else:
        st.warning("⚠️ Please add ground truth questions to run evaluation.")

    # Display results
    if "eval_results" in st.session_state and st.session_state.eval_results:
        st.markdown("---")
        st.subheader("📊 Evaluation Results")
        
        # Show model used
        model_used = st.session_state.get('selected_model', 'Unknown')
        st.info(f"Model used: {model_used}")
        
        # Generate visualizations
        fig, df_eval, metrics = visualize_rag_results(st.session_state.eval_results)
        
        if fig and metrics:
            st.pyplot(fig)
            
            # Display key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Accuracy", f"{metrics['accuracy']:.3f}", f"{metrics['accuracy']*100:.1f}%")
            with col2:
                st.metric("🔍 Precision", f"{metrics['precision']:.3f}", f"{metrics['precision']*100:.1f}%")
            with col3:
                st.metric("📈 Recall", f"{metrics['recall']:.3f}", f"{metrics['recall']*100:.1f}%")
            with col4:
                st.metric("⚖️ F1 Score", f"{metrics['f1_score']:.3f}", f"{metrics['f1_score']*100:.1f}%")
            
            # Detailed results
            st.subheader("📋 Detailed Evaluation Results")
            for i, result in enumerate(st.session_state.eval_results):
                    status_icon = "✅" if result['is_correct'] else "❌"
                    st.write(f"### {status_icon} Question {i+1}")
                    st.write(f"**Query:** {result['query']}")
                    st.write(f"**Reference Answer:** {result['reference']}")
                    st.write(f"**RAG Answer:** {result['prediction']}")
                    
                    with st.expander(f"View Evaluation Details for Q{i+1}"):
                        st.write(f"**LLM Evaluation:** {result['results']}")
                        st.write(f"**Number of Source Documents:** {len(result['docs'])}")
                    st.write("---")
            
            # Download options
            st.subheader("💾 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv_data = df_eval[["query", "reference", "prediction", "results", "is_correct"]].to_csv(index=False)
                st.download_button(
                    "📥 Download Detailed Results (CSV)", 
                    data=csv_data, 
                    file_name=f"rag_evaluation_results_{model_used}.csv", 
                    mime="text/csv"
                )
            
            with col2:
                # Metrics summary download
                metrics_summary = {
                    "model_used": model_used,
                    "total_questions": metrics['total_questions'],
                    "correct_answers": metrics['correct_answers'],
                    "incorrect_answers": metrics['incorrect_answers'],
                    "accuracy": metrics['accuracy'],
                    "precision": metrics['precision'],
                    "recall": metrics['recall'],
                    "f1_score": metrics['f1_score']
                }
                
                st.download_button(
                    "📥 Download Metrics Summary (JSON)",
                    data=json.dumps(metrics_summary, indent=2),
                    file_name=f"rag_metrics_summary_{model_used}.json",
                    mime="application/json"
                )