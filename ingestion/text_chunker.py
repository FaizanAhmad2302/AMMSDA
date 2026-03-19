from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - chunk_overlap

    print(f"✅ Text split into {len(chunks)} chunks")
    return chunks

def chunk_documents(documents):
    """Chunk all documents."""
    all_chunks = []

    for doc in documents:
        print(f"📄 Chunking: {doc['filename']}")
        chunks = chunk_text(doc["content"])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": doc["filename"],
                "chunk_id": i,
                "content": chunk
            })

    print(f"\n✅ Total chunks created: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    from ingestion.pdf_ingestor import load_all_pdfs
    docs = load_all_pdfs()
    chunks = chunk_documents(docs)
    print(f"\n📝 Preview of first chunk:")
    print(chunks[0]["content"])