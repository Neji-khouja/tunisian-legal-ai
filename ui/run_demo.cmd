@echo off
rem Lance l'interface en mode demo : reponses fictives, aucun modele charge, aucun appel reseau.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set RAG_UI_DEMO=1
streamlit run app.py --browser.gatherUsageStats false
