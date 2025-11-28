import time
import os
import re
from typing import List, TypedDict

from llama_cpp import Llama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document


# --- Config ---
LLM_MODEL_PATH = os.path.join(os.getcwd(), "ai-models", "Meta-Llama-3-8B-Instruct.Q5_K_M.gguf")
EMBEDDING_MODEL_PATH = os.path.join(os.getcwd(), "ai-models", "all-MiniLM-L6-v2")
PERSIST_DIRECTORY = "vector_db"


# --- Setup ---
print(f"🚀 Loading embedding model from: {EMBEDDING_MODEL_PATH}")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

print(f"📂 Loading vector database from: {PERSIST_DIRECTORY}")
if not os.path.exists(PERSIST_DIRECTORY):
    print("❌ Vector database not found. Run ingest.py first.")
    exit()

db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
print(f"✅ Loaded database with {db._collection.count()} documents.")

print(f"🧠 Loading local LLM from: {LLM_MODEL_PATH}")
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=8192,
    temperature=0.1,
    max_tokens=1024,
    verbose=False
)
print("✅ LLM loaded successfully.")


# --- Graph State ---
class GraphState(TypedDict):
    question: str
    documents: List[str]
    answer: str


# --- Retrieval Logic ---
def retrieve_docs(state: GraphState):
    print("\n--- NODE: Retrieving documents ---")
    question = state["question"].lower()
    docs = []

    # 1️⃣ Try direct metadata retrieval for any Table/Figure numbers
    matches = re.findall(r"(?:table|figure)\s*(\d+)", question)
    if matches:
        print(f"🎯 Detected direct reference(s): {', '.join(matches)}")
        for num in matches:
            try:
                result = db._collection.get(where={"number": num})
                if result and result.get("documents"):
                    found_docs = [
                        Document(page_content=d, metadata=m)
                        for d, m in zip(result["documents"], result["metadatas"])
                    ]
                    print(f"✅ Found {len(found_docs)} document(s) for Table {num}")
                    docs.extend(found_docs)
            except Exception as e:
                print(f"⚠️ Metadata search failed for Table {num}: {e}")

    # 2️⃣ Fallback to semantic retrieval if no direct matches or for general queries
    if not docs:
        k = 10 if any(x in question for x in ["table", "figure", "section", "appendix"]) else 3
        print(f"🔁 Using semantic retrieval (top {k} chunks)...")
        retriever = db.as_retriever(search_kwargs={"k": k})
        semantic_docs = retriever.invoke(question)
        docs.extend(semantic_docs)

    # Deduplicate and trim overly long content
    seen = set()
    unique_docs = []
    for d in docs:
        text = d.page_content.strip()
        if text and text not in seen:
            seen.add(text)
            unique_docs.append(d)

    print(f"📚 Total retrieved unique chunks: {len(unique_docs)}")
    doc_texts = [doc.page_content for doc in unique_docs]
    return {"question": question, "documents": doc_texts}


# --- Answer Generation ---
def generate_answer(state: GraphState):
    print("\n--- NODE: Gen  erating answer ---")
    question = state["question"]
    docs = state["documents"]

    if not docs:
        return {
            "question": question,
            "documents": [],
            "answer": "That information is not available in the document.",
        }

    # Combine document chunks
    context = "\n\n---\n\n".join(docs)[:8000]

    # Clean and precise instructions
    system_prompt = (
        "You are a precise and factual assistant that extracts information ONLY "
        "from the provided context. The context includes both text and tables.\n"
        "- Use specific details from the context.\n"
        "- If the answer isn't in the context, reply exactly:\n"
        "'That information is not available in the document.'\n"
    )

    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    # ✅ No <|begin_of_text|> — llama_cpp adds this automatically
    formatted_prompt = (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    try:
        response = llm(
            formatted_prompt,
            stop=["<|eot_id|>", "<|end_header_id|>", "</s>"],
            max_tokens=768,  # increase for richer table responses
            temperature=0.1,
        )
        answer = response["choices"][0]["text"].strip()
    except Exception as e:
        answer = f"Error generating answer: {e}"

    print(f"✅ Generated answer: {answer[:120]}...")
    return {"question": question, "documents": docs, "answer": answer}


# --- LangGraph Pipeline ---
print("\n⚙️ Building LangGraph workflow...")
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("generate", generate_answer)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()
print("✅ Graph ready for Q&A.")


# --- Run a sample query ---
if __name__ == "__main__":
    query = "provide info about Administrative costs"
    print(f"\n💬 Ask me anything based on your PDF:\n🧠 You: {query}")

    start = time.time()
    result = app.invoke({"question": query})
    end = time.time()

    print("\n--- 🧾 Result ---")
    print(f"💬 Answer: {result['answer']}")
    print(f"⏱️ Time taken: {end - start:.2f}s")

    docs = result["documents"]
    if docs:
        print("\n📄 Retrieved Context Snippets:")
        for i, d in enumerate(docs[:3], 1):
            snippet = d[:600].replace("\n", " ")
            print(f"--- Doc {i} ---\n{snippet}...\n")
