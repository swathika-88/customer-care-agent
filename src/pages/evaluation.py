import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json
import tempfile
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY1")

# Import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from pages import upload_file, chatbot

def evaluate_rag_pipeline(retriever, ground_truth, api_key=API_KEY, model_name="gpt-4"):
    llm = ChatOpenAI(openai_api_key=api_key, model=model_name)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    results = []

    for item in ground_truth:
        query = item["query"]
        reference_answer = item["reference_answers"]
        result = qa_chain({"query": query})
        prediction = result["result"]
        source_docs = result["source_documents"]

        eval_prompt = f"""
        Evaluate the following:
        Question: {query}
        Reference Answer: {reference_answer}
        Predicted Answer: {prediction}

        Is the predicted answer CORRECT or INCORRECT? Justify briefly and end with "CORRECT" or "INCORRECT".
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

def visualize_rag_results(results):
    if not results:
        return None

    df = pd.DataFrame(results)
    df["Correct"] = df["is_correct"].astype(int)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_data = pd.DataFrame({
        'Category': ['Incorrect', 'Correct'],
        'Count': [sum(df["Correct"] == 0), sum(df["Correct"] == 1)]
    })

    ax.bar(plot_data['Category'], plot_data['Count'], color=["salmon", "skyblue"])
    ax.set_title("RAG Answer Correctness (LLM Judgment)")
    ax.set_ylabel("Count")

    total = len(df)
    for i, row in enumerate(plot_data.itertuples()):
        percentage = 100 * row.Count / total
        ax.text(i, row.Count + 0.1, f"{percentage:.1f}%", ha="center")

    plt.tight_layout()
    return fig, df

def render():
    st.header("RAG System Evaluation")

    if "retriever" not in st.session_state or not st.session_state.retriever:
        st.warning("Please upload and process PDF documents in the 'File Upload' tab first.")
        return

    if not API_KEY:
        st.error("API key not found in environment. Please check your .env file.")
        return

    st.write("Evaluate the RAG system's performance using ground truth data.")

    try:
        from utils.rag_embeddings.groundtruth import ground_truth
        st.success(f"Loaded {len(ground_truth)} ground truth items.")

        with st.expander("Preview Ground Truth Data"):
            for i, item in enumerate(ground_truth):
                st.write(f"**Q{i+1}:** {item['query']}")
                st.write(f"**Reference Answer:** {item['reference_answers']}")
                st.write("---")

    except Exception as e:
        st.error(f"Error loading ground truth data: {e}")
        return

    if "manual_gt" not in st.session_state:
        st.session_state.manual_gt = []

    with st.form("add_qa_pair"):
        query = st.text_area("Question", height=100)
        reference = st.text_area("Reference Answer", height=150)
        if st.form_submit_button("Add QA Pair") and query and reference:
            st.session_state.manual_gt.append({
                "query": query,
                "reference_answers": reference
            })
            st.success("QA pair added!")

    if st.session_state.manual_gt:
        st.write("### Current Ground Truth Items")
        for i, item in enumerate(st.session_state.manual_gt):
            with st.expander(f"QA Pair {i+1}: {item['query'][:50]}..."):
                st.write(f"**Question:** {item['query']}")
                st.write(f"**Reference:** {item['reference_answers']}")
                if st.button(f"Remove Pair #{i+1}"):
                    st.session_state.manual_gt.pop(i)
                    st.rerun()

        if st.button("Save Ground Truth to File"):
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump(st.session_state.manual_gt, f, indent=2)
            with open(f.name, 'r') as f:
                st.download_button("Download JSON", data=f.read(), file_name="ground_truth.json", mime="application/json")

        ground_truth = st.session_state.manual_gt

    if ground_truth and len(ground_truth) > 0:
        if st.button("Run Evaluation"):
            with st.spinner("Evaluating RAG system..."):
                eval_results = evaluate_rag_pipeline(st.session_state.retriever, ground_truth)
                st.session_state.eval_results = eval_results
                st.success("Evaluation complete!")

    if "eval_results" in st.session_state and st.session_state.eval_results:
        st.subheader("Evaluation Results")
        fig, df_eval = visualize_rag_results(st.session_state.eval_results)
        st.pyplot(fig)

        accuracy = df_eval["is_correct"].mean() * 100
        st.metric("Overall Accuracy", f"{accuracy:.1f}%")

        with st.expander("Detailed Evaluation Results"):
            for i, result in enumerate(st.session_state.eval_results):
                st.write(f"### Question {i+1}: {result['query']}")
                st.write(f"**Reference Answer:** {result['reference']}")
                st.write(f"**RAG Answer:** {result['prediction']}")
                st.write(f"**Evaluation:** {result['results']}")
                st.write(f"**Correct:** {'✅' if result['is_correct'] else '❌'}")
                st.write("---")

        csv = df_eval[["query", "reference", "prediction", "results", "is_correct"]].to_csv(index=False)
        st.download_button("Download Results as CSV", data=csv, file_name="rag_evaluation_results.csv", mime="text/csv")
