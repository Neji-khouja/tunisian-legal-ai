@echo off
rem Lance l'interface avec le vrai systeme (index, modeles, cle Groq). A lancer depuis n'importe quel dossier.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set RAG_UI_DEMO=
streamlit run app.py --browser.gatherUsageStats false
