from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

print("Collection count:", db._collection.count())

# Show first few documents directly
docs = db.get()

print("Keys:", docs.keys())

print("Number of documents:", len(docs["documents"]))

if len(docs["documents"]) > 0:
    print("\nFIRST DOCUMENT:")
    print(docs["documents"][0][:300])