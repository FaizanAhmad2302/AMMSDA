import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_ingestor import extract_text_from_pdf
from ingestion.text_chunker import chunk_documents
from ingestion.csv_ingestor import csv_to_text
from embeddings.vector_store import create_embeddings, save_vector_store, load_vector_store, search_similar_chunks
from hypothesis.hypothesis_generator import generate_hypothesis
from knowledge_graph.graph_builder import extract_concepts_from_text, build_knowledge_graph, visualize_graph

def process_inputs_and_query(pdf_files, csv_files, user_query):
    """Main pipeline: ingest multiple PDFs + CSVs, create embeddings, generate hypothesis."""

    if pdf_files is None or len(pdf_files) == 0:
        return "❌ Please upload at least one PDF file!"

    if not user_query.strip():
        return "❌ Please enter a research question!"

    try:
        docs = []

        # Step 1: Extract text from all PDFs
        for pdf_file in pdf_files:
            print(f"📄 Processing PDF: {os.path.basename(pdf_file.name)}")
            text = extract_text_from_pdf(pdf_file.name)
            if text:
                docs.append({
                    "filename": os.path.basename(pdf_file.name),
                    "content": text
                })
                print(f"✅ PDF loaded: {os.path.basename(pdf_file.name)}")

        # Step 2: Load all CSVs if provided
        if csv_files is not None:
            for csv_file in csv_files:
                print(f"📊 Processing CSV: {os.path.basename(csv_file.name)}")
                csv_result = csv_to_text(csv_file.name)
                if csv_result:
                    docs.append({
                        "filename": csv_result["filename"],
                        "content": csv_result["content"]
                    })
                    print(f"✅ CSV loaded: {csv_result['filename']}")

        if not docs:
            return "❌ Could not extract content from uploaded files!"

        print(f"\n📚 Total documents loaded: {len(docs)}")

        # Step 3: Chunk all documents
        chunks = chunk_documents(docs)

        # Step 4: Create embeddings and save
        embeddings = create_embeddings(chunks)
        save_vector_store(chunks, embeddings)

        # Step 5: Generate hypothesis
        hypothesis = generate_hypothesis(user_query)

        return hypothesis

    except Exception as e:
        return f"❌ Error: {str(e)}"

def generate_graph_from_pdfs(pdf_files):
    """Generate knowledge graph from multiple PDFs."""

    if pdf_files is None or len(pdf_files) == 0:
        return None

    try:
        # Combine text from all PDFs
        combined_text = ""
        for pdf_file in pdf_files:
            text = extract_text_from_pdf(pdf_file.name)
            if text:
                combined_text += f"\n\n--- {os.path.basename(pdf_file.name)} ---\n{text}"

        if not combined_text:
            return None

        concepts_data = extract_concepts_from_text(combined_text)
        if not concepts_data:
            return None

        G = build_knowledge_graph(concepts_data)
        visualize_graph(G, title=f"Knowledge Graph - {len(pdf_files)} Paper(s)")

        return "knowledge_graph/knowledge_graph.png"

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Build Gradio UI
with gr.Blocks(title="AMMSDA - Scientific Discovery Agent") as app:

    gr.Markdown("""
    # 🧬 Autonomous Multi-Modal Scientific Discovery Agent
    ### Upload multiple research papers and datasets to get AI-generated hypotheses and knowledge graphs!
    """)

    with gr.Tabs():

        # Tab 1: Hypothesis Generator
        with gr.TabItem("🔬 Hypothesis Generator"):
            gr.Markdown("### Upload multiple research papers and datasets")

            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(
                        label="📄 Upload Research Papers (PDF) - multiple allowed",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )
                    csv_input = gr.File(
                        label="📊 Upload Datasets (CSV) - multiple allowed",
                        file_types=[".csv"],
                        file_count="multiple"
                    )
                    query_input = gr.Textbox(
                        label="🔍 Your Research Question",
                        placeholder="e.g. How can attention mechanisms be improved?",
                        lines=3
                    )
                    submit_btn = gr.Button("🚀 Generate Hypothesis", variant="primary")

                with gr.Column(scale=2):
                    hypothesis_output = gr.Textbox(
                        label="🧬 Generated Hypothesis",
                        lines=20
                    )

            submit_btn.click(
                fn=process_inputs_and_query,
                inputs=[pdf_input, csv_input, query_input],
                outputs=[hypothesis_output]
            )

        # Tab 2: Knowledge Graph
        with gr.TabItem("🕸️ Knowledge Graph"):
            gr.Markdown("### Upload multiple research papers to generate a combined knowledge graph")

            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input_kg = gr.File(
                        label="📄 Upload Research Papers (PDF) - multiple allowed",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )
                    graph_btn = gr.Button("🕸️ Generate Knowledge Graph", variant="primary")

                with gr.Column(scale=2):
                    graph_output = gr.Image(label="📊 Knowledge Graph")

            graph_btn.click(
                fn=generate_graph_from_pdfs,
                inputs=[pdf_input_kg],
                outputs=[graph_output]
            )

        # Tab 3: About
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## About AMMSDA

            **Autonomous Multi-Modal Scientific Discovery Agent** is an AI system that:

            - 📄 **Reads** multiple research papers (PDFs)
            - 📊 **Analyzes** multiple datasets (CSV files)
            - 🧠 **Understands** content using vector embeddings
            - 🔬 **Generates** novel scientific hypotheses
            - 🕸️ **Builds** knowledge graphs linking concepts
            - 💡 **Suggests** experiments to test hypotheses

            ### Tech Stack
            - **LLM**: Groq AI (LLaMA 3.3 70B)
            - **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
            - **Vector DB**: FAISS
            - **Knowledge Graph**: NetworkX + Matplotlib
            - **UI**: Gradio

            ### How to Use
            1. Go to **Hypothesis Generator** tab
            2. Upload one or more research paper PDFs
            3. Optionally upload one or more CSV datasets
            4. Type your research question
            5. Click **Generate Hypothesis**
            6. Go to **Knowledge Graph** tab to visualize concepts
            """)

if __name__ == "__main__":
    print("🚀 Launching AMMSDA...")
    app.launch(share=False)