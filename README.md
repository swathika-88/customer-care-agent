# customer_care_agent
**About**

A conversational AI assistant designed to help Tesla owners get instant answers from Tesla owner manuals.

**Tech Stack**

Language: Python Interface: Streamlit Embeddings: OpenAI / Sentence Transformers Vector Store: FAISS PDF Parsing: PyPDF Architecture: Retrieval-Augmented Generation (RAG)

Project Strcture resources/ └── Tesla owner manual/ # Folder containing Tesla manuals (PDF, TXT, etc.)

src/ ├── utils/ │ └── rag_embeddings/ │ ├── chunking.py # Split documents into chunks │ ├── embedding.py # Generate vector embeddings from text │ ├── vector_store.py # Store embeddings in FAISS │ ├── retriever.py # Retrieve relevant chunks for a user query │ └── generation.py # Generate responses using language models │ ├── pages/ │ ├── init.py │ └── upload_file.py # Streamlit interface for uploading manuals │ ├── chatbot.py # Streamlit interface for the Q&A chatbot | |--- Dockerfile |---docker-compose.yml ├── main.py # Application entry point ├── requirements.txt # Python dependencies └── README.md # Project documentation

**Getting Into the Project To get started with development or exploration, follow these steps:**

Clone the Repository Clone the project from GitHub to your local machine:
clone https://github.com/swathika-88/customer_care_agent.git

Navigate to the Source Code All core application logic and code files are located within the src/ directory:
cd src This is where you’ll find:

main.py: Entry point to launch the Streamlit app

chatbot.py: Chat interface logic

pages/: Streamlit UI modules including file upload

utils/rag_embeddings/: Core RAG pipeline (chunking, embeddings, retrieval, generation)

requirements.txt: All required dependencies

Install Dependencies Install required Python packages from src/requirements.txt:
pip install -r requirements.txt

Configure Environment Variables Set your OpenAI API key by creating a .env file inside the src/ directory:
OPENAI_API_KEY=API_KEY1

Prepare Resources Place Tesla owner manuals (PDFs) inside:
resources/ This content will be processed and indexed for retrieval.

Launch the App From within the src/ directory, run: streamlit run main.py

**Docker**

Create a docker-compose.yml file
Create Dockerfile
To build docker : docker compose up --build
