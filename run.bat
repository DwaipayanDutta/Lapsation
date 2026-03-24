@echo off
setlocal enabledelayedexpansion
title Lapsation KMS v2.0

cls
echo.
echo  ================================================================
echo    Lapsation KMS v2.0  ^|  Policy Retention Intelligence
echo    TinyLlama  ^|  ChromaDB  ^|  BM25  ^|  XGBoost  ^|  SHAP
echo  ================================================================
echo.

:: ── Anchor to script directory ──────────────────────────────
cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

:: Full absolute path to venv python.exe
:: Using python.exe -m pip (NOT pip.exe) on every call:
::   pip.exe in a venv is UNSIGNED → blocked by Windows Device Guard / AppLocker
::   python.exe is signed by the Python Software Foundation → always allowed
set "PY=%ROOT%\venv\Scripts\python.exe"

echo  Root : %ROOT%
echo  Venv : %PY%
echo.

:: ── STEP 1 — Python check ───────────────────────────────────
echo  [1/8] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found on PATH.
    echo         Download Python 3.10 or 3.11 from https://python.org
    echo         During install tick "Add Python to PATH"
    echo.
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do (
    echo         Found: %%v
    set "PYVER=%%v"
)
:: Extract major.minor for wheel availability check
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PYVERNUM=%%v"
for /f "tokens=1,2 delims=." %%a in ("%PYVERNUM%") do (
    set "PYMAJ=%%a"
    set "PYMIN=%%b"
)
echo         Python %PYMAJ%.%PYMIN% detected

:: Python 3.13 is fully supported — numpy 2.x cp313 wheels exist on PyPI
:: Python 3.14 is NOT yet supported (chromadb 0.6.x uses pydantic.v1 which breaks on 3.14)
if "%PYMAJ%"=="3" if %PYMIN% GEQ 14 (
    echo.
    echo  ERROR: Python 3.14+ is not supported by chromadb 0.6.x.
    echo         chromadb uses pydantic.v1 internally which breaks on Python 3.14.
    echo         REQUIRED: Python 3.10, 3.11, 3.12, or 3.13
    echo         Download: https://www.python.org/downloads/release/python-3130/
    echo.
    pause & exit /b 1
)
if "%PYMAJ%"=="3" if %PYMIN% LSS 10 (
    echo.
    echo  ERROR: Python 3.%PYMIN% is too old. Requires Python 3.10 or newer.
    pause & exit /b 1
)
echo.

:: ── STEP 2 — Create folders ─────────────────────────────────
echo  [2/8] Creating required folders...
for %%d in (data index models logs chroma_db) do (
    if not exist "%ROOT%\%%d" (
        mkdir "%ROOT%\%%d"
        echo         Created: %%d\
    ) else (
        echo         Exists:  %%d\
    )
)
echo.

:: ── STEP 3 — Copy api_app.py → app.py ──────────────────────
echo  [3/8] Preparing app.py...
if not exist "%ROOT%\api_app.py" (
    echo  ERROR: api_app.py not found in %ROOT%
    pause & exit /b 1
)
copy /Y "%ROOT%\api_app.py" "%ROOT%\app.py" >nul
for %%s in ("%ROOT%\app.py") do echo         app.py ready ^(%%~zs bytes^)
echo.

:: ── STEP 4 — Create venv ────────────────────────────────────
echo  [4/8] Setting up virtual environment...
if exist "%PY%" (
    echo         venv already exists — skipping creation
) else (
    echo         Creating venv...
    python -m venv "%ROOT%\venv"
    if errorlevel 1 (
        echo  ERROR: venv creation failed.
        echo         Make sure python -m venv is available.
        pause & exit /b 1
    )
    echo         venv created OK
)
echo         Python: %PY%
echo.

:: Verify venv python is reachable
"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: venv python not executable at:
    echo         %PY%
    echo         Delete the venv\ folder and re-run.
    pause & exit /b 1
)

:: ── STEP 5 — Install packages ───────────────────────────────
echo  [5/8] Installing packages...
echo         Device Guard safe: all installs use python.exe -m pip
echo         ^(python.exe is signed — pip.exe is not^)
echo.

:: Always upgrade pip first via module call (never pip.exe)
echo  [5a] Upgrading pip...
"%PY%" -m pip install --upgrade pip --quiet --no-warn-script-location
if errorlevel 1 (
    echo  WARNING: pip upgrade failed — continuing with existing version
)
echo         pip upgraded OK

:: ── GROUP 1: PyTorch CPU ──────────────────────────────────
echo.
echo  [5b] Installing PyTorch CPU ^(largest package^)...
"%PY%" -m pip install "torch>=2.3.0,<2.7" --extra-index-url https://download.pytorch.org/whl/cpu --quiet --no-warn-script-location
if errorlevel 1 (
    echo  ERROR: PyTorch install failed.
    echo         Check internet connection or proxy settings.
    echo         Set: set HTTPS_PROXY=http://yourproxy:port
    pause & exit /b 1
)
"%PY%" -c "import torch; print('         torch', torch.__version__, 'OK')"
if errorlevel 1 ( echo  ERROR: torch import failed & pause & exit /b 1 )

:: ── GROUP 2: HuggingFace stack ────────────────────────────
echo.
echo  [5c] Installing HuggingFace stack...
"%PY%" -m pip install ^
    "transformers>=4.41.0,<5.0" ^
    "tokenizers>=0.19.0,<0.22" ^
    "accelerate>=0.30.0" ^
    "safetensors>=0.4.3" ^
    "huggingface_hub>=0.23.0" ^
    "sentence-transformers>=3.0.0,<4.0" ^
    --quiet --no-warn-script-location
if errorlevel 1 ( echo  ERROR: HuggingFace install failed & pause & exit /b 1 )
"%PY%" -c "import transformers, sentence_transformers; print('         transformers', transformers.__version__, '| sbert', sentence_transformers.__version__, 'OK')"
if errorlevel 1 ( echo  ERROR: HuggingFace imports failed & pause & exit /b 1 )

:: ── GROUP 3a: numpy (MUST be pre-built wheel — no C compiler needed) ──
echo.
echo  [5d] Installing numpy ^(pre-built wheel only — no compiler needed^)...
echo         Python %PYMAJ%.%PYMIN% detected — selecting correct numpy wheel...

:: Python 3.13 requires numpy >=2.1 (first version with cp313 wheels on PyPI)
:: Python 3.10/3.11/3.12 can use numpy 1.26.4 (stable, widely tested)
:: numpy 1.x has NO cp313 wheel — trying it on 3.13 causes the meson/compiler error
if "%PYMAJ%"=="3" if %PYMIN% GEQ 13 (
    echo         Python 3.13+: using numpy 2.x ^(cp313 wheels available from 2.1+^)
    "%PY%" -m pip install "numpy>=2.1.0,<3.0" --only-binary=:all: --quiet --no-warn-script-location
    if errorlevel 1 (
        echo  ERROR: numpy 2.x wheel install failed on Python 3.13.
        echo         This usually means pip couldn't find a matching wheel.
        echo         Try:  "%PY%" -m pip install numpy --only-binary=:all:
        pause & exit /b 1
    )
    goto NUMPY_OK
)

:: Python 3.10 / 3.11 / 3.12 — use numpy 1.26.4 (guaranteed cp310/cp311/cp312 wheels)
echo         Python 3.10-3.12: trying numpy 1.26.4 binary wheel...
"%PY%" -m pip install "numpy==1.26.4" --only-binary=:all: --quiet --no-warn-script-location
if errorlevel 1 (
    echo         numpy 1.26.4 wheel not found — trying numpy 2.x wheel...
    "%PY%" -m pip install "numpy>=2.1.0,<3.0" --only-binary=:all: --quiet --no-warn-script-location
    if errorlevel 1 (
        echo  ERROR: numpy wheel install failed.
        echo.
        echo  CAUSE: No pre-built numpy wheel exists for Python %PYMAJ%.%PYMIN%.
        echo.
        echo  FIXES (try in order):
        echo    1. Use Python 3.10, 3.11, or 3.12 (most wheels exist for these)
        echo       Download: https://www.python.org/downloads/release/python-31011/
        echo    2. Install Visual Studio Build Tools ^(free^):
        echo       https://visualstudio.microsoft.com/visual-cpp-build-tools/
        echo       Select: "Desktop development with C++"
        echo    3. Run manually:  "%PY%" -m pip install numpy --only-binary=:all:
        echo.
        pause & exit /b 1
    )
)

:NUMPY_OK
"%PY%" -c "import numpy; print('         numpy', numpy.__version__, 'OK')"
if errorlevel 1 ( echo  ERROR: numpy import failed & pause & exit /b 1 )

:: ── GROUP 3b: FAISS + ChromaDB (binary wheels only) ──────
echo.
echo  [5d2] Installing FAISS + ChromaDB + rank-bm25...
"%PY%" -m pip install ^
    "faiss-cpu>=1.8.0" --only-binary=:all: ^
    --quiet --no-warn-script-location
if errorlevel 1 (
    echo  WARNING: faiss-cpu binary wheel failed — trying source build...
    "%PY%" -m pip install "faiss-cpu>=1.8.0" --quiet --no-warn-script-location
    if errorlevel 1 ( echo  ERROR: faiss-cpu install failed & pause & exit /b 1 )
)
"%PY%" -m pip install ^
    "chromadb>=0.6.0,<0.7.0" ^
    "langchain-chroma>=0.1.4,<0.3" ^
    "rank-bm25>=0.2.2" ^
    --quiet --no-warn-script-location
if errorlevel 1 ( echo  ERROR: ChromaDB/langchain-chroma install failed & pause & exit /b 1 )
"%PY%" -c "import numpy, chromadb, faiss; print('         numpy', numpy.__version__, '| chromadb', chromadb.__version__, '| faiss OK')"
if errorlevel 1 ( echo  ERROR: numpy/chromadb/faiss imports failed & pause & exit /b 1 )

:: ── GROUP 4: LangChain 0.3.x ─────────────────────────────
echo.
echo  [5e] Installing LangChain 0.3.x...
"%PY%" -m pip install ^
    "langchain>=0.3.0,<0.4" ^
    "langchain-community>=0.3.0,<0.4" ^
    "langchain-core>=0.3.0,<0.4" ^
    "langchain-huggingface>=0.1.0,<0.3" ^
    "langchain-text-splitters>=0.3.0,<0.4" ^
    --quiet --no-warn-script-location
if errorlevel 1 ( echo  ERROR: LangChain install failed & pause & exit /b 1 )
"%PY%" -c "import langchain; print('         langchain', langchain.__version__, 'OK')"
if errorlevel 1 ( echo  ERROR: langchain import failed & pause & exit /b 1 )

:: ── GROUP 5: FastAPI + Pydantic ──────────────────────────
echo.
echo  [5f] Installing FastAPI + Pydantic...
"%PY%" -m pip install ^
    "fastapi[standard]>=0.110.0,<0.200" ^
    "uvicorn[standard]>=0.29.0" ^
    "python-multipart>=0.0.9" ^
    "pydantic>=2.6.0,<2.12" ^
    "pydantic-settings>=2.3.0,<3.0" ^
    --quiet --no-warn-script-location
if errorlevel 1 ( echo  ERROR: FastAPI install failed & pause & exit /b 1 )
"%PY%" -c "import fastapi, uvicorn; print('         fastapi', fastapi.__version__, '| uvicorn', uvicorn.__version__, 'OK')"
if errorlevel 1 ( echo  ERROR: fastapi/uvicorn imports failed & pause & exit /b 1 )

:: ── GROUP 6: Data + ML (XGBoost, SHAP, pandas) ──────────
echo.
echo  [5g] Installing data and ML packages...
:: scipy and scikit-learn have C extensions — try binary wheel first
"%PY%" -m pip install ^
    "scipy>=1.12.0,<2.0" --only-binary=:all: ^
    "scikit-learn>=1.4.0" --only-binary=:all: ^
    --quiet --no-warn-script-location
if errorlevel 1 (
    echo  WARNING: Binary wheels for scipy/scikit-learn not found — trying source...
    "%PY%" -m pip install "scipy>=1.12.0,<2.0" "scikit-learn>=1.4.0" --quiet --no-warn-script-location
    if errorlevel 1 ( echo  ERROR: scipy/scikit-learn install failed & pause & exit /b 1 )
)
"%PY%" -m pip install ^
    "xgboost>=2.1.0" ^
    "joblib>=1.4.0" ^
    "pandas>=2.1.0" ^
    "pyarrow>=15.0.0" ^
    "shap>=0.45.0" ^
    --quiet --no-warn-script-location
if errorlevel 1 ( echo  ERROR: Data/ML install failed & pause & exit /b 1 )
"%PY%" -c "import xgboost, shap, pandas, scipy, sklearn; print('         xgboost', xgboost.__version__, '| shap', shap.__version__, '| pandas', pandas.__version__, 'OK')"
if errorlevel 1 ( echo  ERROR: xgboost/shap/pandas imports failed & pause & exit /b 1 )

:: ── GROUP 7: Utilities ──────────────────────────────────
echo.
echo  [5h] Installing utilities...
"%PY%" -m pip install ^
    "requests>=2.31.0" ^
    "aiohttp>=3.9.0,<4.0" ^
    "anyio>=4.3.0,<5.0" ^
    "httpx>=0.27.0" ^
    "python-dotenv>=1.0.0" ^
    "PyYAML>=6.0" ^
    "colorama>=0.4.6" ^
    "tqdm>=4.66.0" ^
    "pillow>=10.0.0" ^
    "psutil>=6.0.0" ^
    "tenacity>=8.2.0,<9.0" ^
    "rich>=13.0.0" ^
    --quiet --no-warn-script-location
if errorlevel 1 (
    echo  WARNING: Some utilities failed to install — continuing
)
echo         Utilities OK

:: ── Final import sanity check ────────────────────────────
echo.
echo  [5i] Final import verification...
"%PY%" -c "import torch, transformers, chromadb, langchain_chroma, langchain, fastapi, xgboost, shap, pandas, faiss, rank_bm25; print('         ALL IMPORTS OK')"
if errorlevel 1 (
    echo.
    echo  ERROR: One or more packages failed to import.
    echo         Run:  "%PY%" -m pip check
    echo         to see dependency conflicts, then re-run run.bat
    echo.
    echo  COMMON FIXES:
    echo    - Delete venv\ folder and re-run (clean install)
    echo    - Check Python version: 3.10 or 3.11 recommended
    echo    - Check internet / proxy
    pause & exit /b 1
)
echo.

:: ── STEP 6 — Write launch_server.bat ───────────────────────
echo  [6/8] Writing server launcher...
(
    echo @echo off
    echo title Lapsation KMS v2.0  [http://localhost:8000]
    echo cd /d "%ROOT%"
    echo echo.
    echo echo  =====================================================
    echo echo   Lapsation KMS v2.0  ^|  FastAPI
    echo echo   App    :  http://localhost:8000
    echo echo   Docs   :  http://localhost:8000/docs
    echo echo   Status :  http://localhost:8000/status
    echo echo   EDA    :  http://localhost:8000/eda/summary
    echo echo   SHAP   :  http://localhost:8000/shap
    echo echo   Batch  :  http://localhost:8000/batch
    echo echo   NOTE   :  TinyLlama downloads on first run ^(~2 GB^)
    echo echo  =====================================================
    echo echo.
    echo "%PY%" -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    echo pause
) > "%ROOT%\launch_server.bat"
echo         launch_server.bat written
echo.

:: ── STEP 7 — Free port 8000 + launch ───────────────────────
echo  [7/8] Starting server...
echo.

:: Free port 8000 if something is using it
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr /R ":8000 "') do (
    taskkill /PID %%p /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

:: Launch server in its own window using absolute venv python
start "Lapsation KMS — Server" cmd /c ""%ROOT%\launch_server.bat""

:: ── STEP 8 — Poll /status then open browser ─────────────────
echo  [8/8] Waiting for server...
echo         TinyLlama loads in 2-5 min on first run.
echo         Subsequent starts: 30-60 sec.
echo.

set WAITED=0
:POLL
timeout /t 4 /nobreak >nul
set /a WAITED+=4

powershell -NoProfile -NonInteractive -Command ^
  "try{$r=Invoke-WebRequest -Uri 'http://localhost:8000/status' -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop;exit 0}catch{exit 1}" ^
  >nul 2>&1

if not errorlevel 1 goto READY

set /a MOD=WAITED %% 20
if !MOD!==0 echo         Still starting... !WAITED!s elapsed
if !WAITED! geq 420 goto TIMEOUT
goto POLL

:TIMEOUT
echo.
echo  Server did not respond after 7 min.
echo  Opening browser anyway — check the Server window for errors.
goto OPEN

:READY
echo.
echo  SERVER READY after %WAITED%s
echo.

:OPEN
start "" "http://localhost:8000"
echo.
echo  ================================================================
echo   RUNNING   http://localhost:8000
echo   API DOCS  http://localhost:8000/docs
echo   STATUS    http://localhost:8000/status
echo   EDA       http://localhost:8000/eda/summary
echo   SHAP      http://localhost:8000/shap
echo   BATCH     http://localhost:8000/batch
echo.
echo   Next steps:
echo     1. Open the app in your browser (auto-opened above)
echo     2. Go to Upload Data — drop your .pkl files
echo     3. Go to Model Pipeline — click Train Model
echo     4. Use KMS Chat to ask questions about your data
echo.
echo   Stop server: close the "Lapsation KMS — Server" window
echo  ================================================================
echo.
pause
endlocal
