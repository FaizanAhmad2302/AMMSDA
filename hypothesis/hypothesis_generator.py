from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from embeddings.vector_store import load_vector_store, search_similar_chunks

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def generate_hypothesis(user_query):
    """Generate a scientific hypothesis based on research paper content."""

    print(f"\n🔍 Searching relevant context for: '{user_query}'")

    # Load vector store
    index, chunks = load_vector_store()
    if index is None:
        return "❌ No vector store found! Please run embeddings/vector_store.py first."

    # Search relevant chunks
    results = search_similar_chunks(user_query, index, chunks, top_k=5)

    # Build context from top results
    context = ""
    for i, result in enumerate(results):
        context += f"\n[Source {i+1} - {result['chunk']['filename']}]\n"
        context += result["chunk"]["content"] + "\n"

    print("✅ Context retrieved!")
    print("🤖 Generating hypothesis with Groq AI...")

    # Build prompt
    prompt = f"""You are an expert scientific research assistant. 
Based on the following excerpts from research papers, generate a novel scientific hypothesis.

RESEARCH CONTEXT:
{context}

USER QUERY: {user_query}

Please provide:
1. 🔬 HYPOTHESIS: A clear, testable scientific hypothesis
2. 📚 SUPPORTING EVIDENCE: Key points from the research that support this hypothesis
3. 🧪 SUGGESTED EXPERIMENT: A concrete experiment to test this hypothesis
4. 📊 CONFIDENCE SCORE: Your confidence in this hypothesis (0-100%)
5. 💡 EXPLANATION: Brief reasoning behind the hypothesis

Be specific, scientific, and grounded in the provided research context."""

    # Call Groq API
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7
    )

    hypothesis = response.choices[0].message.content
    return hypothesis

if __name__ == "__main__":
    query = "How can attention mechanisms be improved for better performance?"
    result = generate_hypothesis(query)
    print("\n" + "="*60)
    print("🧬 GENERATED HYPOTHESIS")
    print("="*60)
    print(result)