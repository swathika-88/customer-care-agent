from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
def load_and_split_pdf(pdf_file, chunk_size=1000, chunk_overlap=200):
    """
    Load a PDF file and split it into chunks.
    
    Args:
        pdf_file: File object of the PDF
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects
    """
    # Create a temporary file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    # Load the PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    # Remove the temporary file
    os.unlink(tmp_path)
    return chunks