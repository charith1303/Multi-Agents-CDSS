from rag_agent import search_disease

print("Testing RAG...")

result = search_disease(
    "fungal infection"
)

print(result)