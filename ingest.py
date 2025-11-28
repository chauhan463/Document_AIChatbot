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
PDF_PATH = "optimal.pdf"
EMBEDDING_MODEL_PATH = os.path.join(os.getcwd(), "ai-models", "all-MiniLM-L6-v2")
PERSIST_DIRECTORY = "vector_db"


def ingest_pdf(pdf_path, embedding_model_path, persist_directory):
    print(f"🚀 Starting ingestion: {pdf_path}")
    start = time.time()

    if not os.path.exists(pdf_path):
        print("❌ PDF not found.")
        return

    # -------------------------------------------------------------------------
    # 1️⃣ PARSE PDF USING OCR ONLY  (💯 FULLY OFFLINE)
    # -------------------------------------------------------------------------
    print("🔍 Parsing PDF (ocr_only mode)...")
    elements = partition_pdf(
        filename=pdf_path,
        strategy="ocr_only",                # ⭐ FULL OFFLINE
        infer_table_structure=True,
        extract_image_block_to_text=True,
        extract_image_block_types=["Image"],
        languages=["eng"],
    )
    print(f"✅ Extracted {len(elements)} raw elements")

    # -------------------------------------------------------------------------
    # 2️⃣ MERGE CAPTIONS WITH NEXT ELEMENT  (Critical for ocr_only)
    # -------------------------------------------------------------------------
    merged = []
    skip = False

    CAPTION_RE = re.compile(r"(?i)\b(table|figure|fig\.)\s*\d+")

    for i, el in enumerate(elements):
        if skip:
            skip = False
            continue

        text = getattr(el, "text", "").strip()

        # If "Table 3" or "Figure 1" appears, merge with next block
        if CAPTION_RE.search(text) and i + 1 < len(elements):
            next_el = elements[i + 1]
            combined = text + "\n" + getattr(next_el, "text", "").strip()
            el.text = combined
            merged.append(el)
            skip = True
        else:
            merged.append(el)

    print(f"🔗 Caption merge complete: {len(elements)} → {len(merged)}")

    # -------------------------------------------------------------------------
    # 3️⃣ CHUNK BY DOCUMENT STRUCTURE
    # -------------------------------------------------------------------------
    print("📖 Chunking...")
    chunks = chunk_by_title(
        merged,
        max_characters=1200,
        new_after_n_chars=900,
        combine_text_under_n_chars=400,
    )
    print(f"✅ Generated {len(chunks)} chunks")

    # -------------------------------------------------------------------------
    # 4️⃣ CONVERT CHUNKS TO LANGCHAIN DOCUMENTS + METADATA
    # -------------------------------------------------------------------------
    print("🧱 Converting to LangChain Documents...")

    docs = []

    TABLE_RE = re.compile(r"(?i)\btable\s*(\d+)")
    FIG_RE = re.compile(r"(?i)\b(figure|fig\.)\s*(\d+)")

    for chunk in chunks:
        text = getattr(chunk, "text", "").strip()
        if not text:
            continue

        # Extract clean metadata
        if hasattr(chunk.metadata, "to_dict"):
            meta = chunk.metadata.to_dict()
        elif hasattr(chunk.metadata, "as_dict"):
            meta = chunk.metadata.as_dict()
        else:
            meta = dict(chunk.metadata or {})

        # Detect tables
        t = TABLE_RE.search(text)
        if t:
            meta["type"] = "table"
            meta["number"] = t.group(1)

        # Detect figures
        f = FIG_RE.search(text)
        if f:
            meta["type"] = "figure"
            meta["number"] = f.group(2)

        docs.append(Document(page_content=text, metadata=meta))

    print(f"📄 Created {len(docs)} documents with metadata")

    # -------------------------------------------------------------------------
    # 5️⃣ CLEAN METADATA
    # -------------------------------------------------------------------------
    docs = filter_complex_metadata(docs)
    print(f"🧼 Clean documents: {len(docs)}")

    # -------------------------------------------------------------------------
    # 6️⃣ EMBED + STORE IN CHROMA
    # -------------------------------------------------------------------------
    print("🧠 Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)

    print(f"💾 Saving to Chroma DB at {persist_directory}...")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    if hasattr(db, "persist"):
        db.persist()
    elif hasattr(db, "save_local"):
        db.save_local(persist_directory)

    print("\n🎉 Ingestion Complete")
    print(f"📦 Stored chunks: {len(docs)}")
    print(f"⏱ Time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    ingest_pdf(PDF_PATH, EMBEDDING_MODEL_PATH, PERSIST_DIRECTORY)
