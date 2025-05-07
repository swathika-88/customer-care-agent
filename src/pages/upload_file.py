
import os
import streamlit as st
from utils.rag_embeddings.chunking import load_and_split_pdf
from utils.rag_embeddings.vector_store import create_vector_store
from utils.rag_embeddings.retriever import create_retriever
from utils.rag_embeddings.generation import create_generator
import os 
from dotenv import load_dotenv


load_dotenv()

API_KEY1 = os.getenv("API_KEY1")

def render():
    """Render the file upload page"""
    # Title and description
    st.header("Upload PDF Document")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Document Configuration")
        
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.chunks = None
        st.session_state.vector_store = None
        st.session_state.retriever = None
        st.session_state.generator = None
        st.session_state.pdf_name = None
    
    # Document upload section
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Process the uploaded PDF
        if st.button("Process Document") or (st.session_state.pdf_name != uploaded_file.name and uploaded_file is not None):
            if not API_KEY1:
                st.error("Please enter your OpenAI API key in the sidebar")
            else:
                with st.spinner("Processing document..."):
                    # Update the current PDF name
                    st.session_state.pdf_name = uploaded_file.name
                    
                    # Step 1: Load and chunk the PDF
                    st.write("🔄 Chunking document...")
                    chunks = load_and_split_pdf(uploaded_file)
 
                    st.session_state.chunks = chunks
                    st.write(f"✅ Document processed into {len(chunks)} chunks")
                    
                    # Step 2: Create vector store
                    st.write("🔄 Creating vector store...")
                    vector_store = create_vector_store(documents=chunks)
                    st.session_state.vector_store = vector_store
                    st.write("✅ Vector store created")
                    
                    # Step 3: Create retriever
                    st.write("🔄 Setting up retriever...")
                    retriever = create_retriever(vector_store=vector_store)
                    st.session_state.retriever = retriever
                    st.write("✅ Retriever ready")
                    
                    # Step 4: Create generator
                    st.write("🔄 Setting up generator...")
                    generator = create_generator(retriever=retriever)
                    st.session_state.generator = generator
                    st.write("✅ Generator ready")
                    
                    st.session_state.initialized = True
                    st.success("Document processed successfully! You can now ask questions in the Chatbot tab.")
        
        # Display PDF info
        if st.session_state.chunks:
            st.write(f"📄 Loaded document: **{st.session_state.pdf_name}**")
            st.write(f"📊 Total chunks: **{len(st.session_state.chunks)}**")
            
            # Option to clear current document
            if st.button("Clear Document"):
                st.session_state.initialized = False
                st.session_state.chunks = None
                st.session_state.vector_store = None
                st.session_state.retriever = None
                st.session_state.generator = None
                st.session_state.pdf_name = None
                st.experimental_rerun()