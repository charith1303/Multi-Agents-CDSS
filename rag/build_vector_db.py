import pandas as pd

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

df = pd.read_csv("../data/symptom_Description.csv")
print("Rows loaded:", len(df))
print(df.head())

documents = []

for _, row in df.iterrows():

    documents.append(
        Document(
    page_content=f"Disease: {row['Disease']}\nDescription: {row['Description']}",
    metadata={
        "disease": row["Disease"]
    }
)
    )

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory="rag/chroma_db"

)

print("Vector Database Created Successfully")

