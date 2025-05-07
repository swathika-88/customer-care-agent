import streamlit as st
import os
import sys

# Add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import page modules
from pages import upload_file, chatbot

def main():
    """Main application entry point"""
    # Set up Streamlit page config
    st.set_page_config(page_title="Customer Care Assistant", layout="wide")
    
    # Title and description
    st.title("Customer Care Assistant")
    st.markdown("Use the sidebar to navigate between uploading a file and chatting with the assistant.")
    
    # Sidebar: Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["File Upload", "Chatbot"])
    
    # Load the appropriate page
    if page == "File Upload":
        upload_file.render()
    elif page == "Chatbot":
        chatbot.render()

if __name__ == "__main__":
    main()