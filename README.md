# Lore_Spirit

Two-LLM pipeline (Gatekeeper + Lorekeeper) with RAG over cleaned Mage: The Ascension texts.

Purpose: OWLCON 2026 FRIDAY 02/20/2026 "MAGE LARP" 

## Overview
- **Gatekeeper**: safety + rewrite → outputs strict JSON.
- **Lorekeeper**: RAG-only factual answers using provided CONTEXT.
- **Controller**: Python script that serializes Gatekeeper → Lorekeeper → Gatekeeper, with timeouts, retries, and logging.

## Repo Layout
- `Modelfile` (Gatekeeper)
- `Modelfile` (Lorekeeper)  ← consider placing each in its own subfolder: `ollama/gatekeeper/Modelfile`, `ollama/lorekeeper/Modelfile`
- `llm_lore_spirit_controller.py`

*(Data, logs, and local artifacts are ignored via `.gitignore`.)*

## Setup (Windows)
```bash
# 1) Python env
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2) System deps (one-time)
# - Tesseract OCR (Windows installer)
# - Poppler for Windows (add /bin to PATH)
# - Java (Temurin JRE/JDK)
# - Ollama (install & running service)

# 3) Create models
ollama create gatekeeper -f path/to/Modelfile.gatekeeper
ollama create lorekeeper  -f path/to/Modelfile.lorekeeper

# 4) Run controller
python llm_lore_spirit_controller.py
