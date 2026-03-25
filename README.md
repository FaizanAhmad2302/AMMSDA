---
title: AMMSDA
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
---
# 🧬 Autonomous Multi-Modal Scientific Discovery Agent (AMMSDA)

> An AI-powered research assistant that reads multiple scientific papers and datasets, finds hidden connections, and autonomously generates hypotheses, experiment plans, and knowledge graphs.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-orange)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What Problem Does It Solve?

Scientific research is slow and inefficient. Researchers spend months:
- Reading hundreds of papers manually
- Analyzing experimental datasets
- Trying to find connections between different studies
- Forming hypotheses through trial and error

**AMMSDA does all of this in 30 seconds.** ⚡

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| 📄 **Multi-PDF Ingestion** | Upload and process multiple research papers simultaneously |
| 📊 **CSV Dataset Analysis** | Analyze experimental datasets alongside papers |
| 🧠 **Semantic Search** | FAISS vector database finds the most relevant content |
| 🔬 **Hypothesis Generation** | Groq AI generates novel, evidence-backed hypotheses |
| 🕸️ **Knowledge Graphs** | Visual graphs linking concepts across papers |
| 💡 **Experiment Planning** | Suggests concrete experiments to test hypotheses |
| 📊 **Confidence Scoring** | Rates confidence in each generated hypothesis |
| 🖥️ **Interactive UI** | Clean Gradio web interface for easy use |

---

## 🔄 How It Works — Step by Step

```
1. You upload research papers (PDFs) + optional datasets (CSVs)
          ↓
2. System extracts and cleans all text
          ↓
3. Text is split into 500-character overlapping chunks
          ↓
4. Each chunk is converted into vector embeddings
   (using Sentence Transformers all-MiniLM-L6-v2)
          ↓
5. Embeddings stored in FAISS vector database
          ↓
6. You ask a scientific question
          ↓
7. System finds the 5 most relevant chunks using semantic search
          ↓
8. Relevant context sent to Groq AI (LLaMA 3.3 70B)
          ↓
9. AI generates:
   ✅ Novel hypothesis
   ✅ Supporting evidence (with sources)
   ✅ Suggested experiment
   ✅ Confidence score
   ✅ Scientific explanation
          ↓
10. Knowledge Graph visually maps all key concepts
```

---

## 🏗️ Project Structure

```
AMMSDA/
│
├── data/                          # Your PDFs and CSVs go here
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── dataset.csv
│
├── ingestion/                     # Data loading modules
│   ├── pdf_ingestor.py            # Extracts text from PDFs
│   ├── text_chunker.py            # Splits text into chunks
│   └── csv_ingestor.py            # Loads and summarizes CSV datasets
│
├── embeddings/                    # Vector database
│   ├── vector_store.py            # Creates, saves, loads FAISS index
│   ├── index.faiss                # Saved FAISS index (auto-generated)
│   └── chunks.pkl                 # Saved chunks metadata (auto-generated)
│
├── knowledge_graph/               # Graph construction
│   ├── graph_builder.py           # Extracts concepts and builds graph
│   └── knowledge_graph.png        # Saved graph image (auto-generated)
│
├── hypothesis/                    # AI reasoning
│   └── hypothesis_generator.py    # Generates hypotheses using Groq AI
│
├── ui/                            # Web interface
│   └── app.py                     # Gradio UI application
│
├── config.py                      # Configuration and settings
├── .env                           # API keys (never commit this!)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.9+ | Core programming language |
| **LLM** | Groq AI (LLaMA 3.3 70B) | Hypothesis generation & reasoning |
| **Embeddings** | Sentence Transformers | Converting text to vectors |
| **Vector DB** | FAISS | Semantic similarity search |
| **Knowledge Graph** | NetworkX + Matplotlib | Concept visualization |
| **PDF Reading** | PyPDF2 | Extracting text from papers |
| **Data Analysis** | Pandas | CSV dataset processing |
| **UI** | Gradio | Interactive web interface |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9 or higher
- A free Groq API key (get one at [console.groq.com](https://console.groq.com))

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/AMMSDA.git
cd AMMSDA
```

### Step 2: Create a virtual environment
```bash
python -m venv venv
```

### Step 3: Activate the virtual environment
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 4: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Set up your API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Step 6: Run the app
```bash
python ui/app.py
```

Then open **http://127.0.0.1:7860** in your browser!

---

## 📦 Requirements

```
groq
langchain
langchain-community
langchain-groq
openai
faiss-cpu
chromadb
pypdf2
python-dotenv
sentence-transformers
gradio
networkx
matplotlib
pandas
```

Install all with:
```bash
pip install groq langchain langchain-community langchain-groq openai faiss-cpu chromadb pypdf2 python-dotenv sentence-transformers gradio networkx matplotlib pandas
```

---

## 🧪 Demo Usage

### Example 1 — Single Paper
1. Upload `attention_is_all_you_need.pdf`
2. Ask: *"How can attention mechanisms be improved?"*
3. Get a hypothesis with supporting evidence and experiment plan

### Example 2 — Multiple Papers
1. Upload `bert.pdf`, `gpt3.pdf`, `resnet.pdf`, `lora.pdf`
2. Ask: *"What are the key differences between ResNet and Transformer models?"*
3. System finds cross-paper connections and generates a multimodal hypothesis

### Example 3 — Paper + Dataset
1. Upload `attention_is_all_you_need.pdf`
2. Upload `transformer_experiments.csv`
3. Ask: *"What is the optimal number of attention heads?"*
4. AI combines literature + experimental data for a data-driven hypothesis

---

## 📊 Example Output

```
🔬 HYPOTHESIS:
The optimal number of attention heads in a Transformer model is between
8 and 16, as this range balances diverse representation learning with
computational efficiency.

📚 SUPPORTING EVIDENCE:
- Multi-head attention allows the model to jointly attend to information
  from different representation subspaces (Source 1)
- Experimental data shows highest BLEU score (28.4) achieved with
  16 attention heads (Source 2 - CSV dataset)

🧪 SUGGESTED EXPERIMENT:
Train Transformer models with 8, 12, 16, 20, 24 attention heads on
WMT14 English-German dataset. Measure BLEU score and training time.

📊 CONFIDENCE SCORE: 80%

💡 EXPLANATION:
Based on correlation between attention head count and BLEU score in
the experimental data, cross-referenced with theoretical foundations
from the original Transformer paper.
```

---

## 🔮 Future Improvements

- [ ] Support for research paper images and figures
- [ ] Novelty scoring to rate how unique a hypothesis is
- [ ] Auto chart generation from CSV data
- [ ] Web deployment (Hugging Face Spaces / Streamlit Cloud)
- [ ] Chat history and memory across sessions
- [ ] Export hypotheses as PDF reports
- [ ] Support for more file formats (JSON, HDF5)

---

## 🌍 Real World Impact

Similar commercial products in this space:
- **Elicit.ai** — AI research assistant (raised $9M)
- **Consensus.app** — AI for scientific papers
- **SciSpace** — Research paper AI (millions of users)

AMMSDA demonstrates the core technology behind these products, built from scratch as a student project.

---

## 👨‍💻 Author

Built with ❤️ as a portfolio project demonstrating:
- RAG (Retrieval Augmented Generation)
- Vector databases and semantic search
- LLM integration and prompt engineering
- Knowledge graph construction
- Multi-modal AI pipelines
- End-to-end AI product development

---

## 📄 License

MIT License — feel free to use, modify, and share!

---

## ⭐ If you found this useful, please star the repo!

