import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDINGS_DIR

# Load embedding model
print("🔄 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded!")

def create_embeddings(chunks):
    """Convert text chunks into vector embeddings."""
    print(f"\n🔄 Creating embeddings for {len(chunks)} chunks...")
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"✅ Embeddings created! Shape: {embeddings.shape}")
    return embeddings

def save_vector_store(chunks, embeddings):
    """Save embeddings to FAISS index."""
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    # Save FAISS index
    faiss.write_index(index, os.path.join(EMBEDDINGS_DIR, "index.faiss"))

    # Save chunks metadata
    with open(os.path.join(EMBEDDINGS_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Vector store saved! Total vectors: {index.ntotal}")

def load_vector_store():
    """Load FAISS index and chunks from disk."""
    index_path = os.path.join(EMBEDDINGS_DIR, "index.faiss")
    chunks_path = os.path.join(EMBEDDINGS_DIR, "chunks.pkl")

    if not os.path.exists(index_path):
        print("❌ No vector store found! Please run create_embeddings first.")
        return None, None

    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"✅ Vector store loaded! Total vectors: {index.ntotal}")
    return index, chunks

def search_similar_chunks(query, index, chunks, top_k=5):
    """Search for most relevant chunks given a query."""
    query_embedding = model.encode([query])
    distances, indices = index.search(
        np.array(query_embedding, dtype=np.float32), top_k
    )

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append({
                "chunk": chunks[idx],
                "distance": distances[0][i]
            })

    return results

if __name__ == "__main__":
    from ingestion.pdf_ingestor import load_all_pdfs
    from ingestion.text_chunker import chunk_documents

    # Load and chunk documents
    docs = load_all_pdfs()
    chunks = chunk_documents(docs)

    # Create and save embeddings
    embeddings = create_embeddings(chunks)
    save_vector_store(chunks, embeddings)

    # Test search
    print("\n🔍 Testing search...")
    index, chunks = load_vector_store()
    results = search_similar_chunks("attention mechanism transformer", index, chunks)
    print(f"\n📝 Top result preview:")
    print(results[0]["chunk"]["content"][:300])