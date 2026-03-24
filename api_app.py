"""
Lapsation KMS — api_app.py
Integrated:  TinyLlama RAG (ChromaDB + BM25)  |  XGBoost + SHAP  |  build_base pipeline
All PKL data from data/ folder — no PDF dependency.
"""

import os
import sys
import torch
import numpy as np
import pickle
import logging
import shutil
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# langchain-chroma (NOT langchain-community Chroma) — compatible with chromadb <0.7
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_ID      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_DIR     = "models/tinyllama"
DATA_DIR      = "data"           # PKL files only
DB_DIR        = "chroma_db"
INDEX_DIR     = "index"

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 120
VECTOR_K      = 8
BM25_K        = 8
FINAL_K       = 3

EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HTML_FILE    = "lapsation_app.html"

CUST_KEY   = "N_CUST_REF_NO"
POLICY_KEY = "V_POLICY_NO"

MODEL_PATH      = "models/xgb_lapse_model.pkl"
BASE_DATA_PATH  = "models/BASE_MASTER_DATA.pkl"
SHAP_CSV_PATH   = "models/shap_importance.csv"
BATCH_CSV_PATH  = "models/batch_scored.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# LLM
# ─────────────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs("models", exist_ok=True)
        logger.info("Downloading TinyLlama-1.1B (~2 GB) — first run only…")
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
    return MODEL_DIR

def load_llm():
    path = ensure_model()
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to("cpu")
    return tokenizer, model

def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def load_reranker():
    return CrossEncoder(RERANK_MODEL)

# ─────────────────────────────────────────
# PKL LOADER  (failsafe — skip bad files)
# ─────────────────────────────────────────
def load_pkl_data() -> dict:
    loaded = {}
    if not os.path.exists(DATA_DIR):
        logger.warning(f"data/ folder not found — no PKL loaded")
        return loaded
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".pkl"))
    if not files:
        logger.info("No .pkl files in data/ yet")
        return loaded
    for f in files:
        path = os.path.join(DATA_DIR, f)
        try:
            df = pd.read_pickle(path)
            if not isinstance(df, pd.DataFrame):
                logger.warning(f"  SKIP {f} — not a DataFrame")
                continue
            df.columns = df.columns.str.strip()
            loaded[f.replace(".pkl", "")] = df
            logger.info(f"  Loaded {f} | {df.shape[0]:,} rows × {df.shape[1]} cols")
        except Exception as e:
            logger.error(f"  FAILED {f}: {e}")
    logger.info(f"PKL load complete — {len(loaded)} table(s)")
    return loaded

# ─────────────────────────────────────────
# BUILD BASE DATA  (from integrate file)
# Joins all source tables into one master DF
# ─────────────────────────────────────────
def build_base(data: dict) -> Optional[pd.DataFrame]:
    required = ["GNMT_POLICY"]
    missing  = [t for t in required if t not in data and t.lower() not in data]
    if missing:
        logger.warning(f"build_base: missing required table(s) {missing} — skipping join")
        return None

    def get(name):
        return data.get(name) or data.get(name.lower())

    policy         = get("GNMT_POLICY")
    policy_detail  = get("gnmt_policy_detail")
    customer_master= get("gnmt_customer_master")
    customer_addr  = get("gndt_customer_address")
    policy_riders  = get("gnmt_policy_riders")
    receipt        = get("remt_receipt")

    # Concatenate all status log shards
    status_frames = [v for k, v in data.items() if "status_log" in k.lower() or "contract_status" in k.lower()]
    status_log = pd.concat(status_frames, ignore_index=True) if status_frames else None

    df = policy.copy()

    # Policy → Customer key
    if policy_detail is not None:
        try:
            map_df = policy_detail[[POLICY_KEY, CUST_KEY]].drop_duplicates()
            df = df.merge(map_df, on=POLICY_KEY, how="left")
        except Exception as e:
            logger.warning(f"build_base merge policy_detail→cust_key: {e}")

    # Customer master
    if customer_master is not None and CUST_KEY in df.columns:
        try:
            df = df.merge(customer_master, on=CUST_KEY, how="left", suffixes=("", "_cm"))
        except Exception as e:
            logger.warning(f"build_base merge customer_master: {e}")

    # Address (dedup to first per customer)
    if customer_addr is not None and CUST_KEY in df.columns:
        try:
            addr = customer_addr.groupby(CUST_KEY).first().reset_index()
            df = df.merge(addr, on=CUST_KEY, how="left", suffixes=("", "_addr"))
        except Exception as e:
            logger.warning(f"build_base merge address: {e}")

    # Policy detail (dedup)
    if policy_detail is not None:
        try:
            detail = policy_detail.drop_duplicates(subset=[POLICY_KEY])
            df = df.merge(detail, on=POLICY_KEY, how="left", suffixes=("", "_det"))
        except Exception as e:
            logger.warning(f"build_base merge policy_detail: {e}")

    # Riders count
    if policy_riders is not None:
        try:
            rider = policy_riders.groupby(POLICY_KEY).size().reset_index(name="NUM_RIDERS")
            df = df.merge(rider, on=POLICY_KEY, how="left")
        except Exception as e:
            logger.warning(f"build_base merge riders: {e}")

    # Payment counts
    if receipt is not None:
        try:
            pay = receipt.groupby(POLICY_KEY).size().reset_index(name="NUM_PAYMENTS")
            df  = df.merge(pay, on=POLICY_KEY, how="left")
            if CUST_KEY in df.columns:
                cust_pay = receipt.groupby(CUST_KEY).size().reset_index(name="CUST_TOTAL_PAYMENTS")
                df = df.merge(cust_pay, on=CUST_KEY, how="left")
        except Exception as e:
            logger.warning(f"build_base merge receipts: {e}")

    # Status — latest code + transition count
    if status_log is not None:
        try:
            status_log["D_FROM"] = pd.to_datetime(status_log["D_FROM"], errors="coerce")
            latest = status_log.sort_values([POLICY_KEY, "D_FROM"]).groupby(POLICY_KEY).tail(1)
            df = df.merge(latest[[POLICY_KEY, "V_CURR_STAT_CODE"]], on=POLICY_KEY, how="left")
            trans = status_log.groupby(POLICY_KEY).size().reset_index(name="NUM_STATUS_CHANGES")
            df = df.merge(trans, on=POLICY_KEY, how="left")
        except Exception as e:
            logger.warning(f"build_base merge status: {e}")

    # Target
    if "V_CURR_STAT_CODE" in df.columns:
        df["IS_LAPSED"] = (df["V_CURR_STAT_CODE"] == "NB022").astype(int)
    else:
        df["IS_LAPSED"] = 0

    logger.info(f"build_base complete | shape: {df.shape}")
    return df

# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
CANDIDATE_FEATURES = [
    "NUM_PAYMENTS", "NUM_RIDERS", "NUM_STATUS_CHANGES",
    "N_PREM_PAY_TERM", "N_TERM", "CUST_TOTAL_PAYMENTS",
    "NUM_ADDRESSES", "DAYS_TO_DUE", "PAYMENT_FREQ_RATIO", "RIDER_DENSITY"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    if "D_PREM_DUE_DATE" in df.columns:
        try:
            df["D_PREM_DUE_DATE"] = pd.to_datetime(df["D_PREM_DUE_DATE"], errors="coerce")
            df["DAYS_TO_DUE"] = (df["D_PREM_DUE_DATE"] - pd.Timestamp.today()).dt.days
        except Exception:
            df["DAYS_TO_DUE"] = 0

    df["PAYMENT_FREQ_RATIO"] = df.get("NUM_PAYMENTS", pd.Series(0, index=df.index)) / \
                               (df.get("NUM_STATUS_CHANGES", pd.Series(1, index=df.index)) + 1)
    df["RIDER_DENSITY"] = df.get("NUM_RIDERS", pd.Series(0, index=df.index)) / \
                          (df.get("N_TERM", pd.Series(1, index=df.index)) + 1)
    return df

def get_features(df: pd.DataFrame) -> List[str]:
    return [c for c in CANDIDATE_FEATURES if c in df.columns]

# ─────────────────────────────────────────
# RISK LABEL
# ─────────────────────────────────────────
def get_risk_label(p: float) -> str:
    if p >= 0.7:  return "HIGH"
    if p >= 0.4:  return "MEDIUM"
    return "LOW"

# ─────────────────────────────────────────
# BATCH SCORING
# ─────────────────────────────────────────
def batch_score(df: pd.DataFrame, model, features: List[str]) -> pd.DataFrame:
    df = engineer_features(df)
    avail = [c for c in features if c in df.columns]
    X = df[avail].fillna(0)
    df["LAPSE_PROB"] = model.predict_proba(X)[:, 1]
    df["LAPSE_RISK"] = df["LAPSE_PROB"].apply(get_risk_label)
    return df

def monitor_distribution(df: pd.DataFrame) -> dict:
    dist   = df["LAPSE_RISK"].value_counts(normalize=True).round(4).to_dict()
    pstats = df["LAPSE_PROB"].describe().round(4).to_dict()
    return {"risk_distribution": dist, "probability_stats": pstats}

# ─────────────────────────────────────────
# SHAP (optional — graceful if missing)
# ─────────────────────────────────────────
def compute_shap(model, X_sample: pd.DataFrame, features: List[str]) -> Optional[pd.DataFrame]:
    try:
        import shap
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_sample)
        shap_df = pd.DataFrame({
            "feature":    features,
            "importance": np.abs(shap_vals).mean(axis=0)
        }).sort_values("importance", ascending=False)
        shap_df.to_csv(SHAP_CSV_PATH, index=False)
        logger.info(f"SHAP saved → {SHAP_CSV_PATH}")
        return shap_df
    except ImportError:
        logger.info("shap not installed — skipping SHAP analysis")
        return None
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        return None

def shap_explain_row(model, row_df: pd.DataFrame, features: List[str]) -> dict:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        vals      = explainer.shap_values(row_df)[0]
        expl      = dict(zip(features, [float(v) for v in vals]))
        return dict(sorted(expl.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    except Exception:
        return {}

# ─────────────────────────────────────────
# CHROMADB VECTOR STORE  (langchain-chroma, chromadb <0.7)
# ─────────────────────────────────────────
def build_vectordb(chunks, embeddings):
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    vdb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_DIR)
    return vdb

def load_vectordb(embeddings):
    if not os.path.exists(DB_DIR):
        return None
    try:
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    except Exception as e:
        logger.warning(f"Could not load ChromaDB: {e}")
        return None

# ─────────────────────────────────────────
# BM25
# ─────────────────────────────────────────
BM25_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
DOCS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

def build_bm25(chunks):
    return BM25Okapi([c.page_content.lower().split() for c in chunks])

def save_bm25(bm25, chunks):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(BM25_PATH, "wb") as f: pickle.dump(bm25, f)
    with open(DOCS_PATH, "wb") as f: pickle.dump(chunks, f)

def load_bm25_from_disk():
    try:
        with open(BM25_PATH, "rb") as f: bm25   = pickle.load(f)
        with open(DOCS_PATH, "rb") as f: chunks = pickle.load(f)
        return bm25, chunks
    except Exception:
        return None, []

# ─────────────────────────────────────────
# BUILD KMS DOCUMENTS FROM PKL
# Converts each table into text documents
# so TinyLlama can answer data questions
# ─────────────────────────────────────────
def build_docs_from_pkl(data: dict) -> List[Document]:
    docs = []
    for name, df in data.items():
        # Schema
        docs.append(Document(
            page_content=(
                f"TABLE: {name}\n"
                f"Rows: {df.shape[0]:,}  Columns: {df.shape[1]}\n"
                f"Column list: {', '.join(df.columns.tolist())}\n"
                "Data types:\n" + "\n".join(f"  {c}: {t}" for c, t in df.dtypes.astype(str).items())
            ),
            metadata={"source": name, "doc_type": "schema"}
        ))
        # Numeric summary
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            docs.append(Document(
                page_content=f"NUMERIC SUMMARY — {name}\n" + num_df.describe().round(2).to_string(),
                metadata={"source": name, "doc_type": "stats"}
            ))
        # Null rates (>5%)
        null_pct = (df.isnull().mean() * 100).round(2)
        high = null_pct[null_pct > 5]
        if not high.empty:
            docs.append(Document(
                page_content=f"NULL RATES (>5%) — {name}\n" + "\n".join(f"  {c}: {v}%" for c, v in high.items()),
                metadata={"source": name, "doc_type": "nulls"}
            ))
        # Value distributions for categoricals
        for col in df.select_dtypes(include=["object", "category"]).columns[:6]:
            try:
                vc = df[col].value_counts().head(10)
                docs.append(Document(
                    page_content=f"VALUE DISTRIBUTION — {name}.{col}\n" + "\n".join(f"  {v}: {c:,}" for v, c in vc.items()),
                    metadata={"source": name, "doc_type": "distribution", "column": col}
                ))
            except Exception:
                pass
        # Sample rows
        try:
            docs.append(Document(
                page_content=f"SAMPLE ROWS — {name}\n" + df.head(5).fillna("").astype(str).to_string(index=False),
                metadata={"source": name, "doc_type": "sample"}
            ))
        except Exception:
            pass

    # Static lapsation domain knowledge — always included
    docs.append(Document(
        page_content="""LAPSATION MODEL — DOMAIN KNOWLEDGE

Objective:
  Predict which policyholders are likely to lapse so the company can intervene proactively.

Target: IS_LAPSED = 1 when V_CURR_STAT_CODE == 'NB022'

Features:
  NUM_PAYMENTS         — Premium payments made to date
  NUM_RIDERS           — Riders attached to policy
  NUM_STATUS_CHANGES   — Status transition count (instability signal)
  N_PREM_PAY_TERM      — Premium payment term (years)
  N_TERM               — Policy term (years)
  CUST_TOTAL_PAYMENTS  — Customer's total payments across all policies
  DAYS_TO_DUE          — Days to next premium (negative = overdue)
  PAYMENT_FREQ_RATIO   — NUM_PAYMENTS / (NUM_STATUS_CHANGES + 1)
  RIDER_DENSITY        — NUM_RIDERS / (N_TERM + 1)

Model: XGBoost (n_estimators=300, max_depth=5, lr=0.05, scale_pos_weight=3)
SHAP: TreeExplainer provides per-policy top 5 feature contributions

Risk Tiers:  HIGH ≥70%  |  MEDIUM 40-70%  |  LOW <40%

Tables: GNMT_POLICY, gnmt_policy_detail, gnmt_customer_master,
        gndt_customer_address, gnmt_policy_riders, remt_receipt,
        GN_CONTRACT_STATUS_LOG (multiple shards)

Join Keys: V_POLICY_NO, N_CUST_REF_NO

Channels: Agency | DSF | Bancassurance  — Accuracy: 85-92%
""",
        metadata={"source": "domain_knowledge", "doc_type": "domain"}
    ))
    return docs

def chunk_docs(docs):
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)

def rebuild_kms_index(data: dict) -> tuple:
    docs   = build_docs_from_pkl(data)
    chunks = chunk_docs(docs)
    vdb    = build_vectordb(chunks, resources["embeddings"])
    bm25   = build_bm25(chunks)
    save_bm25(bm25, chunks)
    logger.info(f"KMS index rebuilt — {len(chunks)} chunks")
    return vdb, chunks, bm25

# ─────────────────────────────────────────
# HYBRID SEARCH + RERANK
# ─────────────────────────────────────────
def hybrid_search(query, vectordb, bm25, chunks):
    seen, results = set(), []
    for variant in [query, f"lapsation {query}", f"policy {query}"]:
        for d in vectordb.similarity_search(variant, k=VECTOR_K):
            key = (d.metadata.get("source"), d.page_content[:60])
            if key not in seen:
                seen.add(key); results.append(d)
    scores  = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[-BM25_K:][::-1]
    for i in top_idx:
        if i < len(chunks):
            d = chunks[i]
            key = (d.metadata.get("source"), d.page_content[:60])
            if key not in seen:
                seen.add(key); results.append(d)
    return results

def rerank(query, docs, reranker):
    if not docs: return []
    scores = reranker.predict([(query, d.page_content) for d in docs])
    return [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:FINAL_K]]

def build_context(docs):
    ctx, cites = "", []
    for i, doc in enumerate(docs):
        tag  = f"[{i+1}]"
        src  = doc.metadata.get("source", "unknown")
        dtype = doc.metadata.get("doc_type", "")
        ctx  += f"\n{tag} (Table: {src}, type: {dtype})\n{doc.page_content}\n"
        cites.append((tag, src, dtype, doc))
    return ctx, cites

def build_prompt(context, question):
    return (
        "<|system|>\nYou are a Lapsation Analytics expert. Answer questions about "
        "insurance policy lapsation, datasets, model features, and retention strategy. "
        "Use only the provided context.\n"
        f"Context:\n{context}\n</s>\n"
        f"<|user|>\n{question}\n</s>\n<|assistant|>\n"
    )

def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=350, temperature=0.3,
            do_sample=True, top_p=0.85, repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]
    return text.strip()

def is_greeting(text):
    return text.strip().lower() in {"hi","hello","hey","good morning","good afternoon","good evening"}

# ─────────────────────────────────────────
# EDA HELPERS
# ─────────────────────────────────────────
def compute_eda_stats(data: dict) -> dict:
    stats = {}
    for name, df in data.items():
        try:
            num_df = df.select_dtypes(include=[np.number])
            stats[name] = {
                "rows":    int(df.shape[0]),
                "cols":    int(df.shape[1]),
                "columns": df.columns.tolist()[:25],
                "nulls":   int(df.isnull().sum().sum()),
                "numeric_summary": num_df.describe().round(2).to_dict() if not num_df.empty else {}
            }
        except Exception as e:
            stats[name] = {"error": str(e)}
    return stats

# ─────────────────────────────────────────
# PREDICT (single policy)
# ─────────────────────────────────────────
def predict_single(policy_no: str, data: dict, model, features: list, shap_explainer=None) -> dict:
    # Source from base master if built, else from raw GNMT_POLICY
    df_src = resources.get("base_master")
    if df_src is None:
        df_src = data.get("GNMT_POLICY") or data.get("gnmt_policy")
    if df_src is None:
        return {"error": "No policy data available"}

    row_df = df_src[df_src[POLICY_KEY] == policy_no]
    if row_df.empty:
        return {"error": f"Policy '{policy_no}' not found"}

    row = engineer_features(row_df.copy()).iloc[0]
    avail = [c for c in features if c in row.index]
    X    = pd.DataFrame([row[avail]]).fillna(0)

    prob = float(model.predict_proba(X)[0][1])
    risk = get_risk_label(prob)
    top_factors = shap_explain_row(model, X, avail) if shap_explainer is None else {}

    return {
        "policy_no":         policy_no,
        "customer_id":       str(row.get(CUST_KEY, "N/A")),
        "due_date":          str(row.get("D_PREM_DUE_DATE", "N/A")),
        "lapse_probability": round(prob, 4),
        "lapse_risk":        risk,
        "top_factors":       top_factors
    }

# ─────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────
class QueryRequest(BaseModel):
    query:      str
    session_id: str

class CitationSchema(BaseModel):
    tag:     str
    source:  str
    dtype:   str
    snippet: str

class QueryResponse(BaseModel):
    answer:    str
    citations: List[CitationSchema]

class PredictRequest(BaseModel):
    policy_no: str

class TrainRequest(BaseModel):
    run_shap: bool = True
    run_batch: bool = True

# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
app = FastAPI(title="Lapsation KMS", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

resources: Dict[str, Any] = {}

# ── Serve HTML ────────────────────────────
@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists(HTML_FILE):
        with open(HTML_FILE, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h2>lapsation_app.html not found next to app.py</h2>", status_code=404)

# ── Status ────────────────────────────────
@app.get("/status")
async def get_status():
    pkls   = [f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")] if os.path.exists(DATA_DIR) else []
    chunks = resources.get("chunks") or []
    base   = resources.get("base_master")
    return {
        "llm_model":      MODEL_DIR,
        "embeddings":     EMBED_MODEL,
        "reranker":       RERANK_MODEL,
        "vector_store":   "ChromaDB (langchain-chroma)",
        "pkl_datasets":   pkls,
        "total_chunks":   len(chunks),
        "index_ready":    resources.get("vectordb") is not None,
        "model_trained":  resources.get("lapse_model") is not None,
        "base_master_built": base is not None,
        "base_master_rows":  int(len(base)) if base is not None else 0,
        "shap_available":    os.path.exists(SHAP_CSV_PATH),
        "batch_scored":      os.path.exists(BATCH_CSV_PATH),
    }

# ── Upload PKL ────────────────────────────
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)
    saved = []
    for file in files:
        if not file.filename.lower().endswith(".pkl"):
            continue
        dest = os.path.join(DATA_DIR, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)
        logger.info(f"Uploaded: {file.filename}")

    if not saved:
        raise HTTPException(400, "No valid .pkl files uploaded")

    resources["pkl_data"] = load_pkl_data()
    vdb, chunks, bm25 = rebuild_kms_index(resources["pkl_data"])
    resources["vectordb"] = vdb
    resources["chunks"]   = chunks
    resources["bm25"]     = bm25

    return JSONResponse({
        "uploaded":     saved,
        "pkl_tables":   list(resources["pkl_data"].keys()),
        "total_chunks": len(chunks),
    })

# ── EDA Endpoints ─────────────────────────
@app.get("/eda/summary")
async def eda_summary():
    pkl_data = resources.get("pkl_data", {})
    if not pkl_data:
        return JSONResponse({"message": "No PKL data loaded", "tables": {}})
    return JSONResponse({"tables": compute_eda_stats(pkl_data)})

@app.get("/eda/table/{table_name}")
async def eda_table(table_name: str, rows: int = 20):
    pkl_data = resources.get("pkl_data", {})
    if table_name not in pkl_data:
        raise HTTPException(404, f"Table '{table_name}' not found")
    df = pkl_data[table_name]
    return JSONResponse({
        "table":    table_name,
        "shape":    list(df.shape),
        "columns":  df.dtypes.astype(str).to_dict(),
        "sample":   df.head(rows).fillna("").astype(str).to_dict(orient="records"),
        "null_pct": (df.isnull().mean() * 100).round(2).to_dict(),
    })

@app.get("/eda/lapse_dist")
async def lapse_distribution():
    pkl_data = resources.get("pkl_data", {})
    status_tables = [k for k in pkl_data if "status" in k.lower() or "contract" in k.lower()]
    if not status_tables:
        return JSONResponse({"message": "No status log tables found", "data": []})
    try:
        df = pd.concat([pkl_data[t] for t in status_tables], ignore_index=True)
        if "V_CURR_STAT_CODE" in df.columns:
            dist   = df["V_CURR_STAT_CODE"].value_counts().head(15).to_dict()
            lapsed = int(df["V_CURR_STAT_CODE"].eq("NB022").sum())
            total  = int(len(df))
            return JSONResponse({
                "total":  total, "lapsed": lapsed,
                "lapse_rate": round(lapsed / total * 100, 2) if total else 0,
                "status_distribution": [{"status": k, "count": v} for k, v in dist.items()]
            })
    except Exception as e:
        return JSONResponse({"error": str(e), "data": []})
    return JSONResponse({"message": "V_CURR_STAT_CODE not found", "data": []})

# ── Train ─────────────────────────────────
@app.post("/train")
async def train_endpoint(req: TrainRequest = TrainRequest()):
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score
        import joblib

        pkl_data = resources.get("pkl_data", {})
        if not pkl_data:
            raise HTTPException(503, "No PKL data loaded")

        # Build base master
        logger.info("Building base master table…")
        base = build_base(pkl_data)
        if base is None:
            # Fall back to raw GNMT_POLICY
            base = pkl_data.get("GNMT_POLICY") or pkl_data.get("gnmt_policy")
            if base is None:
                raise HTTPException(400, "GNMT_POLICY table not found")

        resources["base_master"] = base
        base.to_pickle(BASE_DATA_PATH)

        # Feature engineering
        base_fe = engineer_features(base)
        features = get_features(base_fe)
        if not features:
            raise HTTPException(400, "No usable feature columns found in base data")

        # Target
        if "IS_LAPSED" not in base_fe.columns:
            if "V_CURR_STAT_CODE" in base_fe.columns:
                base_fe["IS_LAPSED"] = (base_fe["V_CURR_STAT_CODE"] == "NB022").astype(int)
            else:
                raise HTTPException(400, "Cannot determine IS_LAPSED target")

        X = base_fe[features].fillna(0)
        y = base_fe["IS_LAPSED"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        logger.info(f"Training XGBoost on {len(X_train):,} rows, features: {features}")
        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=3, random_state=42, eval_metric="logloss"
        )
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

        resources["lapse_model"]  = model
        resources["feature_cols"] = features

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        acc   = round(float(accuracy_score(y_test, preds)), 4)
        auc   = round(float(roc_auc_score(y_test, proba)), 4)

        result = {
            "status":        "trained",
            "features_used": features,
            "train_rows":    int(len(X_train)),
            "test_rows":     int(len(X_test)),
            "accuracy":      acc,
            "roc_auc":       auc,
            "shap_computed": False,
            "batch_scored":  False,
        }

        # SHAP
        if req.run_shap:
            sample  = X_train.sample(min(1000, len(X_train)), random_state=42)
            shap_df = compute_shap(model, sample, features)
            if shap_df is not None:
                result["shap_top_features"] = shap_df.head(5).to_dict(orient="records")
                result["shap_computed"]     = True

        # Batch scoring
        if req.run_batch and resources.get("base_master") is not None:
            scored = batch_score(resources["base_master"].copy(), model, features)
            scored.to_csv(BATCH_CSV_PATH, index=False)
            dist = monitor_distribution(scored)
            result["batch_scored"]       = True
            result["risk_distribution"]  = dist["risk_distribution"]
            result["probability_stats"]  = dist["probability_stats"]

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Train error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

# ── Predict (single) ──────────────────────
@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    model    = resources.get("lapse_model")
    features = resources.get("feature_cols", [])
    if model is None:
        raise HTTPException(503, "No trained model — run POST /train first")

    result = predict_single(req.policy_no, resources.get("pkl_data", {}), model, features)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return JSONResponse(result)

# ── SHAP importance ───────────────────────
@app.get("/shap")
async def get_shap():
    if not os.path.exists(SHAP_CSV_PATH):
        return JSONResponse({"message": "SHAP not computed yet. Train with run_shap=true.", "data": []})
    df = pd.read_csv(SHAP_CSV_PATH)
    return JSONResponse({"data": df.to_dict(orient="records")})

# ── Batch results ─────────────────────────
@app.get("/batch")
async def get_batch(risk: Optional[str] = None, limit: int = 100):
    if not os.path.exists(BATCH_CSV_PATH):
        return JSONResponse({"message": "No batch results yet. Train with run_batch=true.", "data": []})
    df = pd.read_csv(BATCH_CSV_PATH)
    if risk:
        df = df[df["LAPSE_RISK"] == risk.upper()]
    cols = [c for c in [POLICY_KEY, CUST_KEY, "LAPSE_PROB", "LAPSE_RISK", "D_PREM_DUE_DATE"] if c in df.columns]
    return JSONResponse({
        "total": int(len(df)),
        "data":  df[cols].head(limit).fillna("").astype(str).to_dict(orient="records")
    })

# ── KMS Query ─────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest) -> QueryResponse:
    if is_greeting(request.query):
        return QueryResponse(
            answer="Hello! I'm the Lapsation KMS. Ask me about datasets, lapse model features, table schemas, status codes, or retention strategy.",
            citations=[]
        )
    if not resources.get("vectordb") or not resources.get("chunks"):
        raise HTTPException(503, "KMS index not ready. Upload .pkl files first.")
    try:
        docs           = hybrid_search(request.query, resources["vectordb"], resources["bm25"], resources["chunks"])
        docs           = rerank(request.query, docs, resources["reranker"])
        ctx, cites     = build_context(docs)
        prompt         = build_prompt(ctx, request.query)
        answer         = generate_answer(prompt, resources["tokenizer"], resources["model"])
        return QueryResponse(
            answer=answer,
            citations=[CitationSchema(tag=t, source=s, dtype=d, snippet=doc.page_content[:200])
                       for t, s, d, doc in cites]
        )
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

# ─────────────────────────────────────────
# STARTUP  (failsafe — never crash on missing data)
# ─────────────────────────────────────────
@app.on_event("startup")
async def startup():
    os.makedirs("logs",     exist_ok=True)
    os.makedirs("models",   exist_ok=True)
    os.makedirs("data",     exist_ok=True)
    os.makedirs(INDEX_DIR,  exist_ok=True)

    logger.info("━━━ Lapsation KMS v2.0 starting ━━━")

    # LLM
    try:
        logger.info("Loading TinyLlama-1.1B…")
        resources["tokenizer"], resources["model"] = load_llm()
    except Exception as e:
        logger.error(f"LLM load failed: {e}")
        resources["tokenizer"] = resources["model"] = None

    # Embeddings
    try:
        logger.info("Loading embeddings (all-MiniLM-L6-v2)…")
        resources["embeddings"] = load_embeddings()
    except Exception as e:
        logger.error(f"Embeddings load failed: {e}")
        resources["embeddings"] = None

    # PKL data
    logger.info("Loading PKL datasets…")
    resources["pkl_data"] = load_pkl_data()

    # ChromaDB index — try saved first, rebuild from PKL if stale/missing
    if resources.get("embeddings"):
        vectordb = load_vectordb(resources["embeddings"])
        bm25, chunks = load_bm25_from_disk()

        if vectordb is not None and chunks:
            logger.info(f"Loaded existing ChromaDB index — {len(chunks)} chunks")
            resources["vectordb"] = vectordb
            resources["chunks"]   = chunks
            resources["bm25"]     = bm25
        elif resources["pkl_data"]:
            logger.info("Building KMS index from PKL data…")
            try:
                vdb, chunks, bm25 = rebuild_kms_index(resources["pkl_data"])
                resources["vectordb"] = vdb
                resources["chunks"]   = chunks
                resources["bm25"]     = bm25
            except Exception as e:
                logger.error(f"KMS index build failed: {e}")
                resources["vectordb"] = None
                resources["chunks"]   = []
                resources["bm25"]     = None
        else:
            logger.warning("No PKL data — KMS index not built. Upload .pkl files to enable chat.")
            resources["vectordb"] = None
            resources["chunks"]   = []
            resources["bm25"]     = None

    # Reranker
    try:
        logger.info("Loading reranker…")
        resources["reranker"] = load_reranker()
    except Exception as e:
        logger.error(f"Reranker load failed: {e}")
        resources["reranker"] = None

    # Pre-trained XGB model
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            resources["lapse_model"] = joblib.load(MODEL_PATH)
            logger.info("Loaded pre-trained XGBoost model")
        except Exception as e:
            logger.warning(f"Could not load XGB model: {e}")

    # Pre-built base master
    if os.path.exists(BASE_DATA_PATH):
        try:
            resources["base_master"] = pd.read_pickle(BASE_DATA_PATH)
            logger.info(f"Loaded base master — {len(resources['base_master']):,} rows")
        except Exception as e:
            logger.warning(f"Could not load base master: {e}")

    logger.info("✓ Startup complete — http://localhost:8000")
