import os
import time
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title


# --- CONFIG ---
PDF_PATH = "sample_table.pdf"
EMBEDDING_MODEL_PATH = os.path.join(os.getcwd(), "ai-models", "all-MiniLM-L6-v2")
PERSIST_DIRECTORY = "vector_db"


# --- INGEST FUNCTION ---
def ingest_pdf(pdf_path, embedding_model_path, persist_directory):
    print(f"🚀 Starting ingestion for: {pdf_path}")
    start_time = time.time()

    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: File not found at {pdf_path}")
        return

    # --- STEP 1: Partition PDF (high quality parsing) ---
    print("🔍 Partitioning PDF (detecting text + tables + layout)...")
    elements = partition_pdf(
        filename=pdf_path,
        strategy="ocr_only",               # ✅ keep layout + detect tables properly
        infer_table_structure=True,
        extract_image_block_to_text=True,
        extract_image_block_types=["Image"],
        languages=["eng"],
    )
    print(f"✅ Extracted {len(elements)} base elements.")

    # --- STEP 2: Chunk by title (keeps logical structure) ---
    print("📖 Chunking by title and semantic proximity...")
    chunks = chunk_by_title(
        elements,
        max_characters=1200,
        new_after_n_chars=900,
        combine_text_under_n_chars=400,
    )
    print(f"✅ Created {len(chunks)} logical chunks.")

    # --- STEP 3: Convert to LangChain Documents ---
    print("🧱 Converting chunks to LangChain Document objects...")
    langchain_documents = []
    for chunk in chunks:
        text = getattr(chunk, "text", "").strip()
        meta = {}

        # extract metadata safely
        if hasattr(chunk.metadata, "to_dict"):
            meta = chunk.metadata.to_dict()
        elif hasattr(chunk.metadata, "as_dict"):
            meta = chunk.metadata.as_dict()
        else:
            meta = dict(chunk.metadata or {})

        # ✅ detect table numbers and add correct metadata key for chat.py
        match = re.search(r"(?i)\btable\s*(\d+)", text)
        if match:
            meta["type"] = "table"
            meta["number"] = match.group(1)

        langchain_documents.append(Document(page_content=text, metadata=meta))

    print(f"✅ Converted into {len(langchain_documents)} documents.")

    # --- STEP 4: Clean metadata (remove complex nested types) ---
    filtered_docs = filter_complex_metadata(langchain_documents)
    print(f"✅ Filtered down to {len(filtered_docs)} clean documents.")

    # --- STEP 5: Embed and create vector database ---
    print(f"🧠 Loading embedding model from: {embedding_model_path}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)

    print(f"💾 Creating persistent Chroma database at: {persist_directory}")
    db = Chroma.from_documents(
        documents=filtered_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    # Compatible persistence method for all LangChain versions
    if hasattr(db, "persist"):
        db.persist()
    elif hasattr(db, "save_local"):
        db.save_local(persist_directory)
    else:
        print("⚠️ Warning: Unable to persist Chroma DB (check version compatibility).")

    db = None

    # --- DONE ---
    print("\n✅ Ingestion Complete")
    print(f"📘 Source: {pdf_path}")
    print(f"🧩 Total Chunks Stored: {len(filtered_docs)}")
    print(f"📂 Vector DB: {persist_directory}")
    print(f"⏱ Time: {time.time() - start_time:.2f} sec")


# --- ENTRY POINT ---
if __name__ == "__main__":
    ingest_pdf(PDF_PATH, EMBEDDING_MODEL_PATH, PERSIST_DIRECTORY)
