from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="rag/chroma_db",
    embedding_function=embedding
)

def search_disease(query):

    results = db.similarity_search(
        query,
        k=3
    )

    print("Results Found:", len(results))

    if len(results) == 0:
        return "No matching disease found."

    return results[0].page_content