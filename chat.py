import os
import time
import re
from typing import List, TypedDict

from llama_cpp import Llama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END


# --- CONFIG ---
LLM_MODEL_PATH = os.path.join(os.getcwd(), "ai-models", "Meta-Llama-3-8B-Instruct.Q5_K_M.gguf")
EMBEDDING_MODEL_PATH = os.path.join(os.getcwd(), "ai-models", "all-MiniLM-L6-v2")
PERSIST_DIRECTORY = "vector_db"


# -------------------------------------------------------------------------
# 1️⃣ LOAD EMBEDDINGS + VECTOR DB
# -------------------------------------------------------------------------
print("🚀 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

print("📂 Loading vector DB...")
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
print(f"✅ Loaded DB with {db._collection.count()} documents.")


# -------------------------------------------------------------------------
# 2️⃣ LOAD LOCAL LLM
# -------------------------------------------------------------------------
print("🧠 Loading local LLM...")
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=8192,
    temperature=0.1,
    max_tokens=1024,
    verbose=False,
)
print("✅ LLM Ready!")


class GraphState(TypedDict):
    question: str
    documents: List[str]
    answer: str


# -------------------------------------------------------------------------
# 3️⃣ RETRIEVAL LOGIC
# -------------------------------------------------------------------------
def retrieve_docs(state: GraphState):
    print("\n--- NODE: RETRIEVE ---")
    question = state["question"].lower()

    # Detect direct Table/Figure reference
    direct_match = re.search(r"(table|figure|fig\.)\s*(\d+)", question)
    if direct_match:
        kind, num = direct_match.groups()
        print(f"🎯 Direct reference detected: {kind} {num}")

        results = db._collection.get(where={"number": num})
        if results and results["documents"]:
            print(f"📌 Found {len(results['documents'])} matches")
            return {"question": question, "documents": results["documents"]}

    # fallback vector RAG
    retriever = db.as_retriever(search_kwargs={"k": 8})
    docs = retriever.invoke(question)

    print(f"📚 Retrieved {len(docs)} chunks (fallback).")
    return {"question": question, "documents": [d.page_content for d in docs]}


# -------------------------------------------------------------------------
# 4️⃣ GENERATION LOGIC
# -------------------------------------------------------------------------
def generate_answer(state: GraphState):
    print("\n--- NODE: Generating answer ---")
    question = state["question"]
    docs = state["documents"]
    context = "\n\n---\n\n".join(docs)

    # Strict system message
    system_prompt = (
        "You are a precise and factual assistant. "
        "Use ONLY the provided context. "
        "If the answer is not found, reply exactly:\n"
        "That information is not available in the document."
    )

    # Llama 3 official conversation format
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}\n"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    response = llm(
        prompt,
        max_tokens=800,
        stop=["<|eot_id|>"],
        temperature=0.0,   # 🔥 Fully deterministic
    )

    answer = response["choices"][0]["text"].strip()
    print(f"🧠 Answer: {answer[:120]}...")
    return {"question": question, "documents": docs, "answer": answer}


# -------------------------------------------------------------------------
# 5️⃣ BUILD LANGGRAPH
# -------------------------------------------------------------------------
print("\n⚙️ Building graph...")
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("generate", generate_answer)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()
print("✅ Graph Ready!")


# -------------------------------------------------------------------------
# 6️⃣ RUN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    query = "How many sets are optimal per week"
    print(f"\n🧠 You: {query}")

    start = time.time()
    result = app.invoke({"question": query})
    stop = time.time()

    print("\n--- RESULT ---")
    print("💬", result["answer"])
    print(f"⏱ {stop - start:.2f}s")
