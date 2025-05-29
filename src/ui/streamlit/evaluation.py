import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY1")

def evaluate_rag_pipeline(retriever, questions, api_key=API_KEY, model_name="llama-3.1-8b-instant"):
    """Evaluate RAG pipeline using Groq API."""
    llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    
    results = []
    for item in questions:
        # Get RAG answer
        result = qa_chain({"query": item["query"]})
        prediction = result["result"]
        
        # Evaluate with LLM
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
    """Calculate evaluation metrics."""
    if not results:
        return {}
    
    y_true = [1] * len(results)  # All ground truth are correct
    y_pred = [1 if r["is_correct"] else 0 for r in results]
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'correct': sum(y_pred),
        'total': len(results)
    }

def create_visualizations(results):
    """Create evaluation charts."""
    if not results:
        return None
    
    metrics = calculate_metrics(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Correctness bar chart
    correct_count = metrics['correct']
    incorrect_count = metrics['total'] - correct_count
    
    ax1.bar(['Incorrect', 'Correct'], [incorrect_count, correct_count], 
            color=['salmon', 'lightgreen'])
    ax1.set_title('Answer Correctness')
    ax1.set_ylabel('Count')
    
    # Add percentages
    for i, count in enumerate([incorrect_count, correct_count]):
        pct = 100 * count / metrics['total']
        ax1.text(i, count + 0.1, f"{pct:.1f}%", ha='center')
    
    # Metrics bar chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    metric_values = [metrics['accuracy'], metrics['precision'], 
                    metrics['recall'], metrics['f1_score']]
    
    bars = ax2.bar(metric_names, metric_values, color='lightblue')
    ax2.set_title('Performance Metrics')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    
    # Add values on bars
    for bar, value in zip(bars, metric_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center')
    
    plt.tight_layout()
    return fig, metrics

def render():
    st.header("🔍 RAG System Evaluation")
    st.markdown("Evaluate your RAG system's performance using Groq API")
    
    # Check prerequisites
    if "retriever" not in st.session_state or not st.session_state.retriever:
        st.warning("⚠️ Please upload and process PDF documents first.")
        return
    
    if not API_KEY:
        st.error("❌ Groq API key not found. Please check your .env file.")
        return
    
    # Model selection
    models = ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", 
              "mixtral-8x7b-32768", "gemma-7b-it"]
    selected_model = st.selectbox("Select Model:", models)
    
    # Ground truth management
    st.subheader("📝 Ground Truth Questions")
    
    # Initialize session state
    if "questions" not in st.session_state:
        st.session_state.questions = []
    
    # Add new question
    with st.form("add_question"):
        col1, col2 = st.columns(2)
        with col1:
            query = st.text_area("Question:", height=80)
        with col2:
            answer = st.text_area("Expected Answer:", height=80)
        
        if st.form_submit_button("➕ Add Question") and query and answer:
            st.session_state.questions.append({
                "query": query.strip(),
                "reference_answers": answer.strip()
            })
            st.success("✅ Question added!")
            st.rerun()
    
    # Display current questions
    if st.session_state.questions:
        st.write(f"**Current Questions: {len(st.session_state.questions)}**")
        
        for i, q in enumerate(st.session_state.questions):
            with st.expander(f"Q{i+1}: {q['query'][:50]}..."):
                st.write(f"**Q:** {q['query']}")
                st.write(f"**A:** {q['reference_answers']}")
                if st.button(f"🗑️ Remove", key=f"del_{i}"):
                    st.session_state.questions.pop(i)
                    st.rerun()
        
        # Clear all questions
        if st.button("🗑️ Clear All Questions"):
            st.session_state.questions = []
            st.rerun()
    
    st.markdown("---")
    
    # Run evaluation
    if st.session_state.questions:
        st.subheader("🚀 Run Evaluation")
        st.info(f"Ready to evaluate {len(st.session_state.questions)} questions")
        
        if st.button("▶️ Start Evaluation", type="primary"):
            with st.spinner("🔄 Evaluating... This may take a few minutes."):
                try:
                    results = evaluate_rag_pipeline(
                        st.session_state.retriever, 
                        st.session_state.questions,
                        model_name=selected_model
                    )
                    st.session_state.eval_results = results
                    st.success("✅ Evaluation complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
    else:
        st.warning("⚠️ Please add some questions to evaluate.")
    
    # Display results
    if "eval_results" in st.session_state and st.session_state.eval_results:
        st.markdown("---")
        st.subheader("📊 Results")
        
        # Create visualizations
        fig, metrics = create_visualizations(st.session_state.eval_results)
        
        if fig:
            st.pyplot(fig)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                st.metric("🔍 Precision", f"{metrics['precision']:.3f}")
            with col3:
                st.metric("📈 Recall", f"{metrics['recall']:.3f}")
            with col4:
                st.metric("⚖️ F1 Score", f"{metrics['f1_score']:.3f}")
            
            # Detailed results
            with st.expander("📋 Detailed Results"):
                for i, result in enumerate(st.session_state.eval_results):
                    status = "✅" if result['is_correct'] else "❌"
                    st.write(f"**{status} Q{i+1}:** {result['query']}")
                    st.write(f"**Expected:** {result['reference']}")
                    st.write(f"**RAG Answer:** {result['prediction']}")
                    st.write("---")
            
            # Download results
            df = pd.DataFrame(st.session_state.eval_results)
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                data=csv,
                file_name=f"rag_evaluation_{selected_model}.csv",
                mime="text/csv"
            )