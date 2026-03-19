import re
import json
import networkx as nx
import matplotlib.pyplot as plt
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from embeddings.vector_store import load_vector_store

client = Groq(api_key=GROQ_API_KEY)

def extract_concepts_from_text(text):
    """Use Groq AI to extract key concepts and relationships from text."""
    print("🤖 Extracting concepts with Groq AI...")

    prompt = f"""Analyze this scientific text and extract key concepts and their relationships.

TEXT:
{text[:3000]}

Return ONLY a valid JSON object in this exact format, nothing else:
{{
  "concepts": ["concept1", "concept2", "concept3"],
  "relationships": [
    {{"from": "concept1", "to": "concept2", "relation": "relationship description"}},
    {{"from": "concept2", "to": "concept3", "relation": "relationship description"}}
  ]
}}

Extract 8-12 most important concepts and 8-12 relationships between them."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.3
    )

    raw = response.choices[0].message.content.strip()

    # Extract JSON from response
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        data = json.loads(match.group())
        print(f"✅ Extracted {len(data['concepts'])} concepts and {len(data['relationships'])} relationships!")
        return data
    else:
        print("❌ Could not parse concepts from AI response.")
        return None

def build_knowledge_graph(concepts_data):
    """Build a NetworkX graph from extracted concepts."""
    G = nx.DiGraph()

    # Add nodes
    for concept in concepts_data["concepts"]:
        G.add_node(concept)

    # Add edges
    for rel in concepts_data["relationships"]:
        G.add_edge(rel["from"], rel["to"], label=rel["relation"])

    print(f"✅ Knowledge graph built! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def visualize_graph(G, title="Knowledge Graph"):
    """Visualize the knowledge graph."""
    plt.figure(figsize=(14, 10))
    plt.title(title, fontsize=16, fontweight='bold')

    pos = nx.spring_layout(G, k=2, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='skyblue',
                           node_size=2000, alpha=0.9)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                           arrows=True, arrowsize=20,
                           width=1.5, alpha=0.7)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    short_labels = {k: v[:20] for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, short_labels, font_size=7)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig("knowledge_graph/knowledge_graph.png", dpi=150, bbox_inches='tight')
    print("✅ Knowledge graph saved as knowledge_graph/knowledge_graph.png")
    plt.show()

if __name__ == "__main__":
    from ingestion.pdf_ingestor import load_all_pdfs

    # Load documents
    docs = load_all_pdfs()
    if not docs:
        print("❌ No documents found!")
        exit()

    # Extract concepts
    full_text = docs[0]["content"]
    concepts_data = extract_concepts_from_text(full_text)

    if concepts_data:
        # Build graph
        G = build_knowledge_graph(concepts_data)

        # Visualize
        visualize_graph(G, title="Knowledge Graph - Attention Is All You Need")