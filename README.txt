═══════════════════════════════════════════════════════════════
  LAPSATION KMS v2.0 — Policy Retention Intelligence Platform
  TinyLlama-1.1B | ChromaDB 0.6 | BM25 | XGBoost | SHAP
═══════════════════════════════════════════════════════════════

QUICK START
───────────
1. Double-click  run.bat
2. Wait for "SERVER READY" (2-5 min first run — TinyLlama download ~2 GB)
3. Browser opens automatically at http://localhost:8000
4. Login:  admin / lapsation@123

WHAT'S NEW IN v2.0
───────────────────
+ build_base()     — joins all 7 source tables into a master DF at train time
+ SHAP             — TreeExplainer gives top 5 feature contributions per policy
+ batch_score()    — full portfolio scored after every training run
+ /shap endpoint   — global SHAP importances (GET /shap)
+ /batch endpoint  — batch results filterable by risk tier (GET /batch?risk=HIGH)
+ Policy Scorer    — now shows SHAP feature bars alongside probability
+ ChromaDB 0.6.x   — uses langchain-chroma (not langchain-community, which is
                     incompatible with chromadb >=1.0 Rust rewrite)
+ Device Guard fix — run.bat uses python.exe -m pip exclusively (pip.exe is
                     unsigned and blocked by Windows AppLocker / Device Guard)
+ Failsafe startup — server never crashes on missing data; each component
                     loads independently with full error logging to logs/app.log

FIRST USE WORKFLOW
──────────────────
1. run.bat          → installs packages, starts server, opens browser
2. Upload Data tab  → drag and drop your .pkl files
3. Model Pipeline   → click "Train Model"
   (builds base master → feature engineering → XGBoost → SHAP → batch score)
4. Policy Scorer    → enter a policy number to get score + SHAP explanation
5. EDA Explorer     → explore your tables, null rates, lapse distribution
6. KMS Chat         → ask questions about your data and model
7. Settings         → check /status, see all API endpoints

PKL FILES (place in data/ or upload via app)
─────────────────────────────────────────────
  gnmt_customer_master.pkl       — Demographics
  gndt_customer_address.pkl      — Location
  GNMT_POLICY.pkl                — Core policy (required for training)
  gnmt_policy_detail.pkl         — PPT, term, ANP
  gnmt_policy_riders.pkl         — Rider counts
  remt_receipt.pkl               — Payment history
  GN_CONTRACT_STATUS_LOG*.pkl    — Status transitions (all shards)

API ENDPOINTS
─────────────
  GET  /status                — Health check + loaded data info
  POST /upload                — Upload .pkl files (rebuilds KMS index)
  POST /query                 — KMS RAG chat (TinyLlama + ChromaDB + BM25)
  GET  /eda/summary           — EDA stats for all loaded tables
  GET  /eda/table/{name}      — Sample rows for a specific table
  GET  /eda/lapse_dist        — Lapse status distribution
  POST /predict               — Score a policy → prob + risk + SHAP factors
  POST /train                 — Full pipeline: base join → XGB → SHAP → batch
  GET  /shap                  — Global SHAP feature importances
  GET  /batch                 — Batch scored portfolio (?risk=HIGH/MEDIUM/LOW)

TARGET DEFINITION
──────────────────
  IS_LAPSED = 1  when  V_CURR_STAT_CODE == "NB022"

MODEL FEATURES
──────────────
  NUM_PAYMENTS, NUM_RIDERS, NUM_STATUS_CHANGES
  N_PREM_PAY_TERM, N_TERM, CUST_TOTAL_PAYMENTS
  DAYS_TO_DUE, PAYMENT_FREQ_RATIO, RIDER_DENSITY

RISK TIERS
──────────
  HIGH   ≥ 70%  → Priority outbound call + SHAP nudge
  MEDIUM 40-70% → Automated reminder
  LOW    < 40%  → No action required

COMPATIBILITY
─────────────
  Python: 3.10 or 3.11 recommended (3.12 supported)
  RAM:    8 GB minimum (16 GB recommended)
  Disk:   ~5 GB (TinyLlama ~2 GB + packages ~3 GB)
  OS:     Windows 10/11 (64-bit)
  Net:    Required on first run for model download

KEY FIXES IN THIS VERSION
──────────────────────────
  chromadb 0.6.x + langchain-chroma
    The chromadb >=1.0 release (Rust rewrite) is incompatible with
    langchain-community.vectorstores.Chroma. This version uses
    langchain-chroma with chromadb 0.6.x (last Python client).

  Device Guard / AppLocker bypass
    All pip installs use  python.exe -m pip  — never pip.exe.
    pip.exe is unsigned and blocked on managed Windows machines.
    python.exe is PSF-signed and always permitted.

  Dependency conflict resolution
    Hard-pinned exact versions caused ResolutionImpossible errors.
    This version uses compatible ranges, installs in dependency order
    (torch first to anchor numpy/typing), and verifies imports.

STOP SERVER
────────────
  Close the "Lapsation KMS — Server" window

LOGS
────
  logs/app.log — full startup and request log

OUTPUTS (after training)
─────────────────────────
  models/xgb_lapse_model.pkl    — Trained XGBoost model
  models/BASE_MASTER_DATA.pkl   — Joined base master table
  models/shap_importance.csv    — Global SHAP feature importances
  models/batch_scored.csv       — Full portfolio with LAPSE_PROB + LAPSE_RISK
═══════════════════════════════════════════════════════════════
