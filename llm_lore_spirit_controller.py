

# LORE SPIRIT PROJECT
# CONTROLLER SCRIPT
# 
# Controller skeleton for Gatekeeper/Lorekeeper pipeline with RAG, strict JSON parsing,
# timeouts/retries, detailed logging, and serialization of calls (CPU-friendly).
#
# Save path: /mnt/data/controller_skeleton.py
#
# Notes:
# - Uses the `ollama` Python package if available (pip install ollama).
#   Falls back to HTTP requests if not. Adjust BASE_URL if your Ollama server differs.
# - Optional: Integrate Chroma for retrieval by filling CHROMA_* settings and un-commenting.
# - Logging: writes CSV and JSONL logs under ./logs/ (created automatically).
# - Stop sequences, low temperature, repeat penalty, and small context are set in Modelfiles,
#   but you can also pass overrides in OPTIONS below.
#
# - This is a skeleton: fill in TODOs for your exact RAG retrieval and metadata formatting.

import json
import csv
import time
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, cast
from typing import TYPE_CHECKING

# Only for type checkers (won’t run at runtime)
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
    from chromadb.api.types import QueryResult  # type: ignore


# --- Optional imports for retrieval (uncomment when ready) ---
# from sentence_transformers import SentenceTransformer
# import chromadb
# from chromadb.config import Settings

# Try to import the ollama Python client; if unavailable, fall back to raw HTTP
try:
    import ollama  # type: ignore
    OLLAMA_WITH_CLIENT = True
except Exception:
    OLLAMA_WITH_CLIENT = False
    import requests  # type: ignore

# ---------------------------
# Configuration
# ---------------------------

# Ollama
BASE_URL = "http://127.0.0.1:11434"
MODEL_GATEKEEPER = "gatekeeper"   # The name you used with `ollama create gatekeeper -f Modelfile.gatekeeper`
MODEL_LOREKEEPER = "lorekeeper"   # The name you used with `ollama create lorekeeper  -f Modelfile.lorekeeper`
# Keep-alive hints to reduce reloads during LARP
KEEP_ALIVE = "30m"

# Default options per call. These can mirror your Modelfile; Modelfile settings take precedence.
OPTIONS = {
    "temperature": 0.1,         # Gatekeeper; Lorekeeper is 0.2 in its Modelfile
    "repeat_penalty": 1.15,
    "top_p": 0.9,
    "top_k": 30,
    "num_ctx": 2048,            # Gatekeeper
    "stop": ["```", "<|eot_id|>"]
}

# Timeouts & retries
REQUEST_TIMEOUT_S = 40     # per model call
REQUEST_RETRIES = 1        # how many retries after a failure

# Logging
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CSV_LOG = LOG_DIR / "controller_log.csv"
JSONL_LOG = LOG_DIR / "controller_log.jsonl"

# Retrieval / RAG (fill when ready)
USE_CHROMA = False  # set True when you wire Chroma below
CHROMA_PERSIST_DIR = "./chroma_db"   # Path to existing Chroma collection
CHROMA_COLLECTION = "mage_lore"      # Your collection name
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim default

# If you want to cap lore responses (guard against rambling)
MAX_ANSWER_CHARS = 1200

# ---------------------------
# Utility helpers
# ---------------------------

def sha1_compact(text: str) -> str:
    """Create a compact SHA1 hash for logging without storing raw text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def ensure_csv_header():
    """Ensure CSV log has a header row."""
    header = [
        "timestamp", "step", "model", "input_sha1", "latency_ms",
        "tokens_in", "tokens_out", "result", "reason", "rewrite",
        "retrieval_k", "source_ids", "error"
    ]
    exists = CSV_LOG.exists()
    with CSV_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)


def log_event(step: str, model: str, input_text: str, latency_ms: int,
              tokens_in: Optional[int] = None, tokens_out: Optional[int] = None,
              result: Optional[str] = None, reason: Optional[str] = None, rewrite: Optional[str] = None,
              retrieval_k: Optional[int] = None, source_ids: Optional[List[str]] = None,
              error: Optional[str] = None):
    """Write a log row (CSV + JSONL)."""
    ensure_csv_header()
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "step": step,
        "model": model,
        "input_sha1": sha1_compact(input_text),
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "result": result,
        "reason": reason,
        "rewrite": rewrite,
        "retrieval_k": retrieval_k,
        "source_ids": ",".join(source_ids) if source_ids else None,
        "error": error,
    }
    # CSV
    with CSV_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row[k] for k in [
            "timestamp","step","model","input_sha1","latency_ms","tokens_in","tokens_out",
            "result","reason","rewrite","retrieval_k","source_ids","error"
        ]])
    # JSONL
    with JSONL_LOG.open("a", encoding="utf-8") as jf:
        jf.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_ollama(model: str, prompt: str, options: Dict[str, Any], timeout_s: int, retries: int) -> str:
    """
    Call Ollama to generate a response. Uses `ollama` client if installed, else HTTP.
    Retries on failure; returns the full text response (non-streaming for simplicity).
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            if OLLAMA_WITH_CLIENT:
                # ollama Python client
                resp = ollama.generate(model=model, prompt=prompt, options=options, keep_alive=KEEP_ALIVE)
                # resp: {"model": "...", "created_at": "...", "response": "text", ...}
                return resp.get("response", "")
            else:
                # raw HTTP
                url = f"{BASE_URL}/api/generate"
                payload = {"model": model, "prompt": prompt, "options": options, "keep_alive": KEEP_ALIVE, "stream": False}
                r = requests.post(url, json=payload, timeout=timeout_s)
                r.raise_for_status()
                data = r.json()
                return data.get("response", "")
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.7)  # brief backoff
            else:
                raise e
    # Should never reach; included for completeness
    raise RuntimeError(f"Ollama call failed after retries: {last_err}")


def parse_gatekeeper_json(text: str) -> Dict[str, str]:
    """
    Extract strict JSON object from Gatekeeper output. Be defensive: find the first {...} block.
    Returns dict with keys: result, reason, rewrite.
    Raises ValueError if parsing fails.
    """
    # Try direct parse first
    try:
        obj = json.loads(text.strip())
        return {"result": obj["result"], "reason": obj["reason"], "rewrite": obj["rewrite"]}
    except Exception:
        pass

    # Fallback: find first JSON object using a simple scan
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1]
        obj = json.loads(snippet)
        return {"result": obj["result"], "reason": obj["reason"], "rewrite": obj["rewrite"]}

    raise ValueError("Gatekeeper did not return valid JSON.")


# ---------------------------
# Retrieval (skeleton)
# ---------------------------

def retrieve_context(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Return top-k passages with metadata for Lorekeeper.
    Always returns a list on every path.
    Each item: {"source_id": str, "page": int|None, "text": str, "score": float|None}
    """
    if not USE_CHROMA:
        return [
            {"source_id": "dummy_source", "page": 1, "text": "This is a placeholder context chunk.", "score": None}
        ]

    try:
        # Lazy imports so module loads even if deps are missing until you enable Chroma
        from sentence_transformers import SentenceTransformer  # type: ignore
        import chromadb  # type: ignore
        from chromadb.config import Settings  # type: ignore

        client = chromadb.Client(Settings(persist_directory=CHROMA_PERSIST_DIR))
        collection = client.get_collection(name=CHROMA_COLLECTION)

        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        qvec = embedder.encode([query]).tolist()

        # Chroma returns a QueryResult (TypedDict). Cast to Dict[str, Any] for indexing.
        res_any: Any = collection.query(
            query_embeddings=qvec,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        res: Dict[str, Any] = cast(Dict[str, Any], res_any)

        docs_all = res.get("documents") or []
        metas_all = res.get("metadatas") or []
        dists_all = res.get("distances") or []

        # Ensure first row exists before subscripting
        if not isinstance(docs_all, list) or not docs_all or not docs_all[0]:
            return []

        row_docs: List[str] = docs_all[0]  # list of chunk texts
        row_metas: List[Dict[str, Any]] = metas_all[0] if isinstance(metas_all, list) and metas_all else []
        row_dists: List[float] = dists_all[0] if isinstance(dists_all, list) and dists_all else []

        items: List[Dict[str, Any]] = []
        for idx, doc_text in enumerate(row_docs):
            md: Dict[str, Any] = row_metas[idx] if idx < len(row_metas) and isinstance(row_metas[idx], dict) else {}
            page_val = md.get("page")
            page_int = page_val if isinstance(page_val, int) else None
            score_val = row_dists[idx] if idx < len(row_dists) else None
            items.append({
                "source_id": str(md.get("source_id", "unknown")),
                "page": page_int,
                "text": doc_text or "",
                "score": score_val,
            })
        return items

    except Exception:
        # Optional: log_event("retrieval_error", "chroma", query, 0, error=str(e))
        return []


def format_context(passages: List[Dict[str, Any]]) -> str:
    """
    Format retrieval results for Lorekeeper prompt.
    """
    lines = []
    for p in passages:
        sid = p.get("source_id", "unknown")
        page = p.get("page", "")
        prefix = f"[S:{sid} p:{page}]" if page != "" else f"[S:{sid}]"
        lines.append(f"{prefix} {p.get('text','').strip()}")
    return "\n".join(lines)


# ---------------------------
# Pipeline
# ---------------------------

def handle_user_query(user_input: str, retrieval_k: int = 5) -> str:
    """
    Serialized flow:
      1) Gatekeeper (pre-check) -> JSON {result, reason, rewrite}
      2) If APPROVED: retrieve context and ask Lorekeeper
      3) Gatekeeper (post-check) on Lorekeeper's answer
      4) Return final or rejection
    """
    # 1) Gatekeeper pre-check
    t0 = time.time()
    gk_raw = call_ollama(MODEL_GATEKEEPER, user_input, OPTIONS, REQUEST_TIMEOUT_S, REQUEST_RETRIES)
    t1 = time.time()
    latency_ms = int((t1 - t0) * 1000)
    try:
        gk = parse_gatekeeper_json(gk_raw)
        log_event("gate_in", MODEL_GATEKEEPER, user_input, latency_ms,
                  result=gk["result"], reason=gk["reason"], rewrite=gk["rewrite"])
    except Exception as e:
        log_event("gate_in", MODEL_GATEKEEPER, user_input, latency_ms, error=f"parse_error: {e}")
        return "Sorry, I couldn't process your request at this time."

    if gk["result"].upper() != "APPROVED":
        return f"(Gatekeeper) {gk['reason']}"

    # 2) RAG retrieval for Lorekeeper
    passages = retrieve_context(gk["rewrite"], k=retrieval_k)
    ctx = format_context(passages)
    lore_prompt = f"USER_QUERY: {gk['rewrite']}\n\nCONTEXT:\n{ctx}"
    source_ids = [f"{p.get('source_id','unknown')}@{p.get('page','')}" for p in passages]

    # 3) Lorekeeper answer
    t2 = time.time()
    lk_raw = call_ollama(MODEL_LOREKEEPER, lore_prompt, OPTIONS, REQUEST_TIMEOUT_S, REQUEST_RETRIES)
    t3 = time.time()
    latency_ms2 = int((t3 - t2) * 1000)
    # Optionally cap runaway responses defensively
    if MAX_ANSWER_CHARS and len(lk_raw) > MAX_ANSWER_CHARS:
        lk_raw = lk_raw[:MAX_ANSWER_CHARS] + " …"

    log_event("lore", MODEL_LOREKEEPER, lore_prompt, latency_ms2,
              retrieval_k=retrieval_k, source_ids=source_ids)

    # 4) Gatekeeper post-check
    t4 = time.time()
    gk2_raw = call_ollama(MODEL_GATEKEEPER, f"Check this response:\n{lk_raw}", OPTIONS, REQUEST_TIMEOUT_S, REQUEST_RETRIES)
    t5 = time.time()
    latency_ms3 = int((t5 - t4) * 1000)
    try:
        gk2 = parse_gatekeeper_json(gk2_raw)
        log_event("gate_out", MODEL_GATEKEEPER, lk_raw, latency_ms3,
                  result=gk2["result"], reason=gk2["reason"], rewrite=gk2["rewrite"])
    except Exception as e:
        log_event("gate_out", MODEL_GATEKEEPER, lk_raw, latency_ms3, error=f"parse_error: {e}")
        # Fail-open or fail-closed? Here we fail-closed for safety.
        return "The response was withheld due to a formatting issue."

    if gk2["result"].upper() != "APPROVED":
        return f"(Gatekeeper) Response withheld. {gk2['reason']}"

    # Optionally return gk2["rewrite"], but we usually keep Lorekeeper's answer intact.
    return lk_raw


# ---------------------------
# Demo entry point
# ---------------------------

if __name__ == "__main__":
    # Minimal demo query. During LARP, your UI/CLI would pass actual user text here.
    sample = "What is the difference between Arete and Quintessence?"
    print(">>> USER:", sample)
    try:
        final_answer = handle_user_query(sample, retrieval_k=5)
        print(">>> LORE SPIRIT:", final_answer)
    except Exception as e:
        print("Controller error:", e)



'''
TO-DO list:
1.  For the function retrieve_context(): Wire the function to Chroma
    when embeddings are ready. For now, returns a stub.
2. Add logging (logging module) [Original user input; substituted (blacklist/length) or not;
    final prompt used; Lorekeeper’s response; whether Gatekeeper approved it]
3. Add more robust error handling
4. Save all exchanges to a log file for monitoring during game and later post-game review
5. Move BLACKLIST, MAX_WORDS, etc., into a separate config.py file for easier adjustment and reuse
6. GUI interface (themed appropriately to Mage: the Ascension setting)
'''

