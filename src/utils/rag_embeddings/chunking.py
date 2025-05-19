from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
def load_and_split_pdf(file_obj, chunk_size=1000, chunk_overlap=200, metadata=None):
    """
    Load and split a PDF file into chunks.
    
    Args:
        file_obj: A file-like object containing the PDF
        chunk_size: The size of each chunk in characters
        chunk_overlap: The overlap between chunks in characters
        metadata: Additional metadata to add to each chunk
        
    Returns:
        A list of document chunks
    """

    is_file_like = hasattr(file_obj, 'read') # Check if it's a file-like object
    # Create a temporary file to save the uploaded file
    if is_file_like:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                  tmp_file.write(file_obj.read())
                  tmp_path = tmp_file.name
    else:
         tmp_path = file_obj
    
    # Load the PDF
         loader = PyPDFLoader(tmp_path)
         documents = loader.load()
        
        # Add source metadata to each document
         if metadata:
            for doc in documents:
                # Add provided metadata while preserving existing metadata
                if not doc.metadata:
                    doc.metadata = {}
                # For each page, add both the source file and the page number
                doc.metadata.update(metadata)
                # Make sure page number is included if not already
                if 'page' not in doc.metadata:
                    doc.metadata['page'] = doc.metadata.get('page', 0)
        
        # Create text splitter
         text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Split documents
         chunks = text_splitter.split_documents(documents)
        
        # Ensure each chunk has proper metadata
         for i, chunk in enumerate(chunks):
            # Add chunk number for tracking
             chunk.metadata['chunk_id'] = i
            # Make sure each chunk has source info
             if metadata and 'source' in metadata:
                chunk.metadata['source'] = metadata['source']
        
    return chunks
    
