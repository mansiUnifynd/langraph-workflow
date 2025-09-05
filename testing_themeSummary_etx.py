import os
import uuid
import asyncio
from pathlib import Path
from typing import TypedDict, List, Dict, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import FakeEmbeddings
from langgraph.checkpoint.memory import MemorySaver

# ========== CONFIG ==========
USER_ID = "user-123"

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    model="moonshotai/kimi-k2:free",
    temperature=0.7
)

# Initialize embedding model (uncomment real embeddings for production; using fake for dev)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = FakeEmbeddings(size=1352)

# Initialize vector store
vector_store = InMemoryVectorStore(embedding=embeddings)

# ========== STATE ==========
class ThemeState(TypedDict, total=False):
    files: List[Dict[str, str]]       # {"path": str, "content": str}
    summaries: List[Dict[str, str]]   # {"path": str, "summary": str}
    verification_results: List[Dict[str, str]]  # {"path": str, "status": str}
    search_result: Optional[Dict[str, Any]]  # {"path": str, "summary": str, "memory_id": str}

# ========== UTILITY FUNCTIONS ==========
async def upsert_memory(
    content: str,
    context: str,
    memory_id: Optional[uuid.UUID] = None,
    user_id: str = USER_ID
):
    """Upsert a memory in the vector store."""
    mem_id = str(memory_id or uuid.uuid4())
    
    global vector_store
    if vector_store is None:
        raise ValueError("Vector store not initialized")
    
    doc = Document(
        page_content=content,
        metadata={"memory_id": mem_id, "user_id": user_id, "context": context}
    )
    
    try:
        vector_store.add_documents([doc], ids=[mem_id])
        return mem_id
    except Exception as e:
        print(f"Error upserting memory {mem_id}: {e}")
        return None

async def retrieve_summary_by_path(context: str, user_id: str = USER_ID) -> Optional[Dict[str, str]]:
    """Retrieve the summary for a given file path from the vector store."""
    global vector_store
    if vector_store is None:
        raise ValueError("Vector store not initialized")
    
    try:
        # Use a dummy query to retrieve documents and filter by metadata
        dummy_query = "check"
        results = await vector_store.asimilarity_search(
            query=dummy_query,
            k=100  # Large k to ensure we get all documents
        )
        for doc in results:
            if (doc.metadata.get("context") == context and 
                doc.metadata.get("user_id") == user_id):
                return {
                    "path": doc.metadata["context"],
                    "summary": doc.page_content,
                    "memory_id": doc.metadata["memory_id"]
                }
                
        return None
    except Exception as e:
        print(f"Error retrieving summary for {context}: {e}")
        return None

async def check_summary_in_memory(context: str, user_id: str = USER_ID) -> bool:
    """Check if a summary for the given context exists in the vector store."""
    result = await retrieve_summary_by_path(context, user_id)
    return result is not None

# ========== NODES ==========

async def load_theme(state: ThemeState):
    theme_dir = "/Users/macbookair-unifynd/langgraph-workflow/shopify-theme"
    files = []
    for file in Path(theme_dir).rglob("*.liquid"):
        content = file.read_text(encoding="utf-8", errors="ignore")
        files.append({"path": str(file), "content": content})
    print(f"Loaded {len(files)} files from {theme_dir}")
    return {"files": files}

async def summarize_and_store(state: ThemeState):
    summaries = []
    print(">>> summarize_and_store called with", len(state.get("files", [])), "files")
    for f in state["files"]:
        print("Summarizing file:", f["path"])
        try:
            prompt = f"Summarize Shopify theme file {f['path']}:\n{f['content'][:3000]}"
            msg = await llm.ainvoke([{"role": "user", "content": prompt}])
            print("LLM returned:", msg)
            summaries.append({"path": f["path"], "summary": msg.content})
        except Exception as e:
            print(f"Error processing file {f['path']}: {e}")
    print("Summaries collected:", summaries)
    return {"summaries": summaries}

async def store_summaries(state: ThemeState):
    summaries = state.get("summaries", [])
    print(">>> store_summaries called with", len(summaries), "summaries")
    for s in summaries:
        try:
            result = await upsert_memory(
                content=s["summary"],
                context=s["path"],
                user_id=USER_ID
            )
            if result:
                print(f"\nStored summary for {s['path']} (Memory ID: {result}):")
                print(f"Summary content:\n{s['summary']}\n")
            else:
                print(f"Failed to store summary for {s['path']}")
        except Exception as e:
            print(f"Error storing summary for {s['path']}: {e}")
    return {"summaries": summaries}

async def verify_summaries(state: ThemeState):
    summaries = state.get("summaries", [])
    verification_results = []
    print(">>> verify_summaries called with", len(summaries), "summaries")
    for s in summaries:
        try:
            exists = await check_summary_in_memory(context=s["path"], user_id=USER_ID)
            status = "Stored" if exists else "Not found"
            verification_results.append({"path": s["path"], "status": status})
            print(f"Verification for {s['path']}: {status}")
        except Exception as e:
            print(f"Error verifying summary for {s['path']}: {e}")
            verification_results.append({"path": s["path"], "status": f"Error: {str(e)}"})
    return {"verification_results": verification_results}

async def search_summary(state: ThemeState):
    """Node to ask for user query and search for similar summary in vector store."""
    query = input("Enter a query describing the theme file: ").strip()
    
    global vector_store
    if vector_store is None:
        raise ValueError("Vector store not initialized")
    
    try:
        results = await vector_store.asimilarity_search(query=query, k=1)
        if results:
            doc = results[0]
            if doc.metadata.get("user_id") == USER_ID:
                path = doc.metadata.get("context")
                summary = doc.page_content
                memory_id = doc.metadata.get("memory_id")
                print(f"\nFound similar file:\nPath: {path}\nSummary: {summary}\nMemory ID: {memory_id}\n")
                return {"search_result": {"path": path, "summary": summary, "memory_id": memory_id}}
            else:
                print("No matching summary found for user.")
                return {"search_result": None}
        else:
            print("No results found.")
            return {"search_result": None}
    except Exception as e:
        print(f"Error during search: {e}")
        return {"search_result": None}

# ========== BUILD WORKFLOW ==========
builder = StateGraph(ThemeState)

builder.add_node("load_theme", load_theme)
builder.add_node("summarize_and_store", summarize_and_store)
builder.add_node("store_summaries", store_summaries)
builder.add_node("verify_summaries", verify_summaries)
builder.add_node("search_summary", search_summary)

builder.add_edge("__start__", "load_theme")
builder.add_edge("load_theme", "summarize_and_store")
builder.add_edge("summarize_and_store", "store_summaries")
builder.add_edge("store_summaries", "verify_summaries")
builder.add_edge("verify_summaries", "search_summary")
builder.add_edge("search_summary", END)

graph = builder.compile()

# ========== RUN WORKFLOW ==========
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "theme-processing-thread"}}
    result = asyncio.run(graph.ainvoke({}, config=config))
    print("\nFinal Result:")
    print(result)
    print("\nVerification Results:")
    for vr in result.get("verification_results", []):
        print(f"Path: {vr['path']}, Status: {vr['status']}")
    search_res = result.get("search_result")
    if search_res:
        print("\nSearch Result:")
        print(f"Path: {search_res['path']}")
        print(f"Summary: {search_res['summary']}")
        print(f"Memory ID: {search_res['memory_id']}")
    else:
        print("\nNo search result found.")