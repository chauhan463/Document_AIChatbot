import os
import time
import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

# Unstructured imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# --- Offline Environment Setup ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["UNSTRUCTURED_HF_HUB_OFFLINE"] = "1"
os.environ["UNSTRUCTURED_LOCAL_MODELS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSTRUCTURED_DISABLE_TELEMETRY"] = "1"
os.environ["YOLOX_MODEL_PATH"] = os.path.join("models", "unstructured", "yolo_x_layout", "yolox_l0.05.onnx")

# --- Config ---
PDF_PATH = "sample_table.pdf"
EMBEDDING_MODEL_PATH = os.path.join(os.getcwd(), "ai-models", "all-MiniLM-L6-v2")
PERSIST_DIRECTORY = "vector_db"


def ingest_pdf(pdf_path, embedding_model_path, persist_directory):
    print(f"🚀 Starting ingestion for: {pdf_path}")
    start_time = time.time()

    # --- Validate Paths ---
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: PDF file not found at {pdf_path}")
        sys.exit(1)

    if not os.path.exists(embedding_model_path):
        print(f"❌ ERROR: Embedding model not found at {embedding_model_path}")
        sys.exit(1)

    if not os.path.exists(os.environ["YOLOX_MODEL_PATH"]):
        print(f"❌ ERROR: Local YOLOX model not found at {os.environ['YOLOX_MODEL_PATH']}")
        sys.exit(1)

    # --- 1. Partition PDF (hi_res + table aware + offline) ---
    print("🔍 Partitioning PDF with Unstructured (hi_res, offline mode)...")
    try:
        elements = partition_pdf(
            filename=pdf_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_images_in_pdf=True,
            ocr_languages="eng",  # Works with PaddleOCR or Tesseract
        )
    except Exception as e:
        print(f"❌ ERROR: Failed to partition PDF.\n{e}")
        print("💡 Tip: Ensure unstructured[pdf,hi_res] + tesseract or paddleocr are installed and available offline.")
        return

    if not elements:
        print("❌ ERROR: No elements extracted from PDF.")
        return
    print(f"✅ Extracted {len(elements)} elements from document.")

    # --- 2. Chunk Elements ---
    print("📖 Chunking elements by title...")
    chunks = chunk_by_title(
        elements,
        max_characters=1000,
        new_after_n_chars=800,
        combine_text_under_n_chars=400,
    )
    print(f"✅ Split into {len(chunks)} chunks.")

    # --- 3. Convert to LangChain Documents ---
    print("🧱 Converting chunks to LangChain Document objects...")
    langchain_documents = [
        Document(
            page_content=chunk.text.strip(),
            metadata=chunk.metadata.to_dict() if hasattr(chunk, "metadata") else {}
        )
        for chunk in chunks if chunk.text.strip()
    ]

    # --- 4. Filter Metadata ---
    print("🧹 Cleaning up metadata...")
    filtered_documents = filter_complex_metadata(langchain_documents)
    print(f"✅ Filtered down to {len(filtered_documents)} clean documents.")

    # --- 5. Embeddings (Local HuggingFace, Offline) ---
    print(f"🧠 Loading local embedding model: {embedding_model_path}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={"device": "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"},
        )
    except Exception as e:
        print(f"❌ ERROR loading embeddings: {e}")
        return

    # --- 6. Create Vector Store ---
    print(f"💾 Creating vector store in: {persist_directory}")
    db = Chroma.from_documents(
        documents=filtered_documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    # --- 7. Persist Offline Vector DB ---
    print("📦 Saving database locally...")
    try:
        if hasattr(db, "persist"):
            db.persist()
        else:
            db._client.persist()  # For newer versions
    except Exception as e:
        print(f"⚠️ Warning: persist() not found or failed. {e}")

    db = None  # close connection

    end_time = time.time()
    print("\n✅ Ingestion Complete")
    print(f"📘 PDF: {pdf_path}")
    print(f"🧩 Total Chunks: {len(filtered_documents)}")
    print(f"📂 Saved to: {persist_directory}")
    print(f"⏱ Time: {end_time - start_time:.2f} sec")
    print("🚫 Fully offline — no internet calls made ✅")


if __name__ == "__main__":
    ingest_pdf(PDF_PATH, EMBEDDING_MODEL_PATH, PERSIST_DIRECTORY)
