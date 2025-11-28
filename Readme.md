🚀 Local PDF Question-Answering System (Offline AI Chatbot)
<p align="center"> <img src="https://img.shields.io/badge/AI%20Chatbot-Offline%20LLM-blue?style=for-the-badge" /> <img src="https://img.shields.io/badge/PDF%20QA-Automated%20Extraction-green?style=for-the-badge" /> <img src="https://img.shields.io/badge/Llama3-GGUF-orange?style=for-the-badge" /> <img src="https://img.shields.io/badge/Chroma-VectorDB-purple?style=for-the-badge" /> </p>

A fully offline, privacy-preserving, and highly accurate PDF Question-Answering system powered by:

Meta Llama-3 (GGUF)

Unstructured (OCR + table/figure extraction)

Chroma VectorDB

HuggingFace Embeddings

LangGraph Retrieval Pipeline

It supports text, tables, figures, and section-based queries with metadata-aware retrieval for perfect accuracy.

🌟 Features
🔐 100% Offline – No Internet Required

All models run locally (GGUF + sentence-transformers). No data leaves your machine.

📄 PDF Intelligence

Extracts and understands:

Text paragraphs

Tables (Table 1, Table 7, Table 12…)

Figures (Figure 1, Fig. 2, etc.)

Sections / subsections

Captions + layout relationships

🎯 Metadata-Aware Retrieval

Understands direct queries such as:

Tell me about Table 11.
Explain Figure 1.
What does Section 3.2 discuss?

⚡ Fast, Optimized Workflow

Smart chunking

OCR fallback

Vector search + direct metadata search

High-accuracy LLM answers

🧠 System Architecture
                 ┌────────────────────┐
                 │      training.pdf   │
                 └──────────┬─────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │   Unstructured OCR    │
                │ (text + tables + figs)│
                └──────────┬───────────┘
                           │ elements
                           ▼
                ┌──────────────────────┐
                │   Chunk by Title     │
                │ (semantic grouping)  │
                └──────────┬───────────┘
                           │ chunks
                           ▼
                ┌──────────────────────┐
                │ Metadata Injection   │
                │  {type: table, number: 5} │
                └──────────┬───────────┘
                           │ documents
                           ▼
                ┌──────────────────────┐
                │   Embeddings (HF)    │
                └──────────┬───────────┘
                           │ vectors
                           ▼
                ┌──────────────────────┐
                │     ChromaDB         │
                └──────────┬───────────┘
                           │ retrieve
                           ▼
                ┌──────────────────────┐
                │     LangGraph        │
                │  (Retrieve + LLM)    │
                └──────────┬───────────┘
                           │ context
                           ▼
                ┌──────────────────────┐
                │   Llama-3 GGUF       │
                │  Local Answer Engine │
                └──────────────────────┘

📁 Folder Structure
ai-chatbot/
│
├── ingest.py              # Build vector DB from PDFs
├── chat.py                # Ask questions interactively
│
├── ai-models/             # Place GGUF + embedding models here (ignored by Git)
│   └── .placeholder
│
├── training/              # Place your PDFs here
│   └── .placeholder
│
├── vector_db/             # Auto-created (ignored)
│
├── utils/                 # Optional helper utilities
│
├── requirements.txt
├── .gitignore
└── README.md

🛠 Installation
1️⃣ Clone the repository
git clone https://github.com/<yourname>/ai-chatbot.git
cd ai-chatbot

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Download local models (manually)

Place inside ai-models/:

Meta-Llama-3-8B-Instruct.Q5_K_M.gguf
all-MiniLM-L6-v2/

4️⃣ Add your PDFs
training/
└── training.pdf

5️⃣ Run ingestion
python ingest.py

6️⃣ Ask questions
python chat.py

🧪 Example Queries
What does Table 4 say?
Summarize Figure 2.
Explain the CONSORT diagram (Figure 1).
What are the results in Section 3?
List all tables.
Tell me about Table 11, 12, and 13.

📊 Sample Output

Example for a table query:

You: Tell me about Table 7

Answer:
Table 7 summarizes the non-current assets for the years 2008–2010...


Example for a figure query:

You: Explain Figure 1

Answer:
Figure 1 is a CONSORT flow diagram showing how participants moved...

⚙️ Technologies Used
Component	Purpose
Unstructured	PDF parsing, OCR, table/figure detection
ChromaDB	Vector database
HuggingFace Embeddings	Semantic vector encoding
Llama-3 (GGUF via llama.cpp)	Local LLM inference
LangGraph	Retrieval + answer pipeline
Python	Orchestrating everything
🔥 Advanced Features
✔ Metadata Injection

Every table/figure is labeled:

{
  "type": "table",
  "number": "7"
}

✔ Direct metadata lookup

For fast, precise matching.

✔ OCR fallback

Even scanned PDFs are processed.

✔ Vector similarity fallback

Handles text-based questions.

⚠️ Notes

This repo does not include any models due to size.

Please place GGUF + embedding models manually in ai-models/.

👨‍💻 Contributing

Pull requests are welcome!
Feel free to add:

UI (Streamlit / Gradio)

Support for images as embeddings

Multi-PDF knowledge bases

📜 License

MIT License – free to use, modify, and distribute.

❤️ Support

If you like this project, please ⭐ star the repository!
