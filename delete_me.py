import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize the Embedder (turns text into vectors)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Initialize ChromaDB (saves to disk)
client = chromadb.PersistentClient(path="./ghire_tm")
collection = client.get_or_create_collection(name="RAG_project")

# 3. Add Data
texts = ["The sky is blue.", "Python is a programming language."]
# Pre-compute embeddings instead of letting Chroma do it internally (Faster!)
embeddings = embedder.encode(texts).tolist()

collection.add(
    ids=["1", "2"],
    documents=texts,
    embeddings=embeddings
)

# 4. Query Data
query_text = "What color is the sky?"
query_embedding = embedder.encode([query_text]).tolist()

# Search the database
results = collection.query(
    query_embeddings=query_embedding,
    n_results=1
)

print(results['documents']) 
# Output: [['The sky is blue.']]