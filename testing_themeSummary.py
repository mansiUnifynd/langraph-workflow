import os
import uuid
import asyncio
from pathlib import Path
from typing import TypedDict, List, Dict, Optional

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


# ========= CONFIG =========
USER_ID = "user-456"

# Setup
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"



# Initialize model (via OpenRouter)
llm = ChatOpenAI(
    model="moonshotai/kimi-k2:free",
    temperature=0.7,
)



# ========= STATE =========
class ThemeState(TypedDict, total=False):
    files: List[Dict[str, str]]       # {"path": str, "content": str}
    summaries: List[Dict[str, str]]   # {"path": str, "summary": str}


# ========= NODES =========
async def load_theme(state: ThemeState, config: Optional[dict] = None):
    """Load all Shopify theme files into memory."""
    theme_dir = "/Users/macbookair-unifynd/langgraph-workflow/shopify-theme"
    files = []
    for file in Path(theme_dir).rglob("*.liquid"):
        content = file.read_text(encoding="utf-8", errors="ignore")
        files.append({"path": str(file), "content": content})
    print(f"📂 Loaded {len(files)} files from {theme_dir}")
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



async def upsert_memory(state: ThemeState, config: RunnableConfig, store: BaseStore):
    """Upsert a memory into LangGraph's store (persists across sessions)."""
    # user_id = config["configurable"]["user_id"]
    user_id = USER_ID
    namespace = ("memory", user_id)
    key = str(uuid.uuid4())

    summaries = state.get("summaries", [])
    print("Summaries to store:", summaries)
    await store.aput(namespace, key, {"memory": summaries})

    return {"memory_id": key}


# ========= BUILD WORKFLOW =========
async def setup_graph():
    builder = StateGraph(ThemeState)

    builder.add_node("load_theme", load_theme)
    builder.add_node("upsert_memory", upsert_memory, store=True, config=True)
    builder.add_node("summarize_and_store", summarize_and_store)


    builder.add_edge("__start__", "load_theme")
    builder.add_edge("load_theme", "summarize_and_store")
    builder.add_edge("summarize_and_store", "upsert_memory")
    builder.add_edge("upsert_memory", END)


    # Use MemorySaver so Studio can persist memory between runs
    return builder.compile()


# Expose the graph for LangGraph Studio
graph = asyncio.run(setup_graph())


