
import os
import streamlit as st
from utils.rag_embeddings.chunking import load_and_split_pdf
from utils.rag_embeddings.vector_store import create_vector_store
from utils.rag_embeddings.retriever import create_hybrid_retriever
from utils.rag_embeddings.generation import create_generator
from utils.rag_embeddings.embedding import embeddings
from dotenv import load_dotenv

load_dotenv()

API_KEY1 = os.getenv("API_KEY1")

def render():
    """Render the file upload page for multiple PDFs"""
    # Title and description
    st.header("Upload PDF Documents")
    st.write("Upload one or more PDF files to create a knowledge base")
    
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
        st.session_state.pdf_names = []
        st.session_state.uploaded_files = []
    
    # Document upload section - changed to accept multiple files
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    # Process uploaded PDFs
    new_files = False
    current_file_names = [file.name for file in uploaded_files] if uploaded_files else []
    
    # Check if new files were added or removed
    if set(current_file_names) != set(st.session_state.pdf_names):
        new_files = True
        st.session_state.uploaded_files = uploaded_files
    
    if uploaded_files:
        # Display information about uploaded files
        st.write(f"📚 {len(uploaded_files)} file(s) selected")
        for file in uploaded_files:
            st.write(f"- {file.name}")
        
        # Process button
        if st.button("Process Documents") or new_files:
            if not API_KEY1:
                st.error("Please enter your OpenAI API key in the sidebar")
            else:
                with st.spinner("Processing documents..."):
                    # Update the current PDF names
                    st.session_state.pdf_names = current_file_names
                    
                    # Process each PDF and collect all chunks
                    all_chunks = []
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    total_files = len(uploaded_files)
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        file_name = uploaded_file.name
                        st.write(f"🔄 Processing: {file_name}")
                        
                        # Step 1: Load and chunk the PDF
                        chunks = load_and_split_pdf(
                            uploaded_file, 
                            # Include source filename in metadata
                            metadata={"source": file_name}
                        )
                        
                        st.write(f"✅ {file_name} processed into {len(chunks)} chunks")
                        all_chunks.extend(chunks)
                        
                        # Update progress bar
                        progress_bar.progress((i + 1) / total_files)
                    
                    st.session_state.chunks = all_chunks
                    st.write(f"✅ All documents processed into {len(all_chunks)} total chunks")
                    
                    # Step 2: Create embeddings from all chunks
                    st.write("🔄 Computing embeddings...")
                    embed_docs = embeddings(all_chunks)
                    st.session_state.embeddings = embed_docs
                    st.write("✅ Embeddings created")
                    
                    # Step 3: Create vector store using pre-computed embeddings
                    st.write("🔄 Creating vector store...")
                    vector_store = create_vector_store(
                        documents=all_chunks,
                        embed_docs=embed_docs
                    )
                    st.session_state.vector_store = vector_store
                    st.write("✅ Vector store created")
                    
                    # Step 3: Create retriever
                    st.write("🔄 Setting up retriever...")
                    retriever = create_hybrid_retriever(vector_store=vector_store)
                    st.session_state.retriever = retriever
                    st.write("✅ Retriever ready")
                    
                    # Step 4: Create generator
                    st.write("🔄 Setting up generator...")
                    generator = create_generator(retriever=retriever)
                    st.session_state.generator = generator
                    st.write("✅ Generator ready")
                    
                    st.session_state.initialized = True
                    st.success("Documents processed successfully! You can now ask questions in the Chatbot tab.")
        
        # Display PDF info
        if st.session_state.chunks:
            st.write("## Knowledge Base Info")
            st.write(f"📚 Loaded documents: **{len(st.session_state.pdf_names)}**")
            
            # Create an expander to show document names
            with st.expander("View Document Names"):
                for name in st.session_state.pdf_names:
                    st.write(f"- {name}")
            
            st.write(f"📊 Total chunks in knowledge base: **{len(st.session_state.chunks)}**")
            
            # Option to clear current documents
            if st.button("Clear All Documents"):
                st.session_state.initialized = False
                st.session_state.chunks = None
                st.session_state.vector_store = None
                st.session_state.retriever = None
                st.session_state.generator = None
                st.session_state.pdf_names = []
                st.session_state.uploaded_files = []
                
    else:
        st.info("Please upload one or more PDF files to begin.")
        
        # Clear state if no files are uploaded
        if st.session_state.pdf_names:
            st.session_state.pdf_names = []
            st.session_state.uploaded_files = []
            