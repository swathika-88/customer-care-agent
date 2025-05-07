from sentence_transformers import SentenceTransformer

def embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_docs = model.encode([doc.page_content for doc in chunks])
    return embed_docs
