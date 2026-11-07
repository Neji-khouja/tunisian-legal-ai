# Interface Streamlit — recherche dans les codes tunisiens

Interface de conversation en français pour les deux systèmes du projet :

- **Vérifié** (v2, par défaut) : l'agent de `agent/agent_graph.py`. Il analyse la question, peut demander une précision
  ou refuser une question hors champ, cherche les articles, rédige des affirmations citées, puis vérifie chaque
  affirmation contre l'article cité. Les étapes s'affichent pendant le calcul. La conversation garde le fil (questions
  de suivi, réponse à une demande de précision).
- **Rapide** (v1.1) : `pipeline.ask_question`, une recherche puis une réponse en une étape, sans vérification
  affirmation par affirmation et sans mémoire de la conversation.

L'interface appelle les mêmes points d'entrée que le harnais d'évaluation (`eval/run_answer_eval.py`) : le graphe
compilé `agent_graph.app` (diffusé par `stream`, avec un identifiant de fil par session) et `pipeline.load_vectorstore`
+ `pipeline.ask_question`. Aucun fichier de `agent/` ni du pipeline n'est modifié.

## Lancer en local (Windows, invite de commandes `cmd`)

Prérequis : l'environnement conda `rag-app` (`requirements.lock.txt`, Streamlit compris), l'index `db_v1_1/` à la racine
du dépôt, les modèles dans le cache Hugging Face, et une clé Groq dans `.env` (`GROQ_API_KEY=...`) ou saisie dans
l'interface.

```bat
conda activate rag-app
cd C:\Users\deadx\OneDrive\Documents\Projects\RAG_PDF\ui
run.cmd
```

`run.cmd` fixe `PYTHONIOENCODING=utf-8`, `HF_HUB_OFFLINE=1` et `TRANSFORMERS_OFFLINE=1`, puis lance
`streamlit run app.py`. Le navigateur s'ouvre sur http://localhost:8501. Le premier message charge l'index et les
modèles (environ une demi-minute).

## Mode démo (sans modèle, sans réseau)

```bat
conda activate rag-app
cd C:\Users\deadx\OneDrive\Documents\Projects\RAG_PDF\ui
run_demo.cmd
```

ou `set RAG_UI_DEMO=1` puis `streamlit run app.py` (ou `streamlit run app.py -- --demo`). Le mode démo n'importe ni
l'agent ni le pipeline et n'envoie aucune requête. Il renvoie des résultats fictifs, clairement signalés, pour chaque
cas : réponse vérifiée, demande de précision (puis réponse au message suivant), refus hors champ, vérification échouée
(bandeau « Réponse non vérifiée »), réponse en arabe (de droite à gauche), limite de quota et erreur inattendue. Des
boutons d'exemples apparaissent dans le panneau latéral. `RAG_UI_DEMO_DELAY` règle la pause entre deux étapes
(0,35 s par défaut). Les captures de `ui/screenshots/` ont été prises dans ce mode (Playwright, Chromium) : accueil,
réponse vérifiée, demande de précision puis réponse, refus hors champ, vérification échouée, arabe, limite de quota et
mode Rapide.

## Tests

```bat
conda activate rag-app
cd C:\Users\deadx\OneDrive\Documents\Projects\RAG_PDF
set PYTHONIOENCODING=utf-8
python ui\tests\test_ui.py
```

Les tests n'utilisent ni réseau ni Groq. Ils couvrent les vues de chaque cas, les messages d'erreur, la politique de
clé, la clé par session (avec des sessions simultanées), et l'application complète en mode démo
(`streamlit.testing`).

## Clés API : politique

**Une clé par session est possible sans modifier `agent/`.** `agent/analyze.py` crée un seul client `ChatGroq` à
l'import, avec la clé de l'environnement, et tous les nœuds le réutilisent ; `pipeline.py` et `translate.py` créent
leurs clients à partir de `config.GROQ_API_KEY`. Tous finissent dans le SDK `groq`, qui signe chaque requête par la
propriété `auth_headers` du client. `ui/keys.py` remplace cette propriété, au niveau de la classe, par une version qui
lit d'abord la clé de la session courante (une `ContextVar`), sinon la clé du client (celle du propriétaire).

- **Isolation entre sessions.** Streamlit exécute chaque session dans son propre fil, et LangGraph exécute ses nœuds
  avec une copie du contexte de l'appelant : la clé d'une session ne sert qu'aux requêtes de cette session. Un test
  le vérifie avec quatre sessions simultanées.
- **Limite.** Le procédé repose sur un détail interne du SDK `groq` (la propriété `auth_headers`, version 0.37.1
  figée dans `requirements.lock.txt`). Après une mise à jour du SDK, relancer `ui\tests\test_ui.py` : le test de clé par
  session échoue si ce détail change.

Réglages (variables d'environnement, lues au démarrage) :

| Variable | Valeurs | Effet |
|---|---|---|
| `UI_OWNER_KEY` | `on` (défaut) / `off` | Sans clé personnelle, répondre avec la clé du propriétaire (`GROQ_API_KEY` de `.env`). |
| `UI_OWNER_DAILY_CAP` | entier, `0` = sans limite (défaut) | Nombre de questions par jour sur la clé du propriétaire, toutes sessions confondues. |

- **Clé personnelle.** Champ « Votre clé Groq » du panneau latéral : elle est utilisée pour les questions de la session
  seulement, gardée en mémoire le temps de la session, jamais écrite ni journalisée. Le bouton « Oublier ma clé »
  l'efface. Elle passe avant la clé du propriétaire et n'est pas comptée dans sa limite.
- **Clé du propriétaire.** Le compteur quotidien vit dans la mémoire du processus : il repart à zéro à minuit (heure
  locale) ou au redémarrage. Une question est comptée avant son exécution, même si elle échoue ensuite.
- **Sans clé du tout** (pas de `GROQ_API_KEY` et pas de clé personnelle), l'interface refuse la question avec un
  message, sans rien envoyer.

## Confidentialité

- Les questions et les réponses ne sont ni écrites sur disque ni journalisées. La conversation vit dans
  `st.session_state`. Pour les questions de suivi du mode Vérifié, elle vit aussi dans la mémoire en RAM de l'agent
  (`MemorySaver`), que « Nouvelle conversation » libère pour le fil concerné. Un fil abandonné sans ce bouton reste en
  mémoire jusqu'au redémarrage du processus.
- Les journaux des modules qui écrivent la requête au niveau INFO (`pipeline` écrit « Search query ») sont limités au
  niveau WARNING. Les erreurs vont à la console sous forme de type d'exception et d'emplacement dans le code, sans le
  message (un message d'erreur du fournisseur peut citer la question ou la sortie du modèle).
- Aucune statistique : `ui/.streamlit/config.toml` désactive `gatherUsageStats`, et `run.cmd` passe aussi l'option.
- **Les questions sont envoyées à Groq** (fournisseur du modèle) pour être traitées : c'est la seule sortie de données.
- L'agent écrit encore « Analyze failed: … » dans la console quand l'analyse échoue (ligne `print` de
  `agent/agent_graph.py`, non modifiée) : le message d'exception du fournisseur, pas la question.

## Mémoire et disque (mesurés le 2026-11-07 sur la machine de développement)

| Élément | Taille |
|---|---|
| Index `db_v1_1/` (Chroma, 6031 *chunks*) | 74 Mo sur disque |
| Modèle d'*embedding* `OrdalieTech/Solon-embeddings-large-0.1` | 2,2 Go dans le cache Hugging Face |
| *Reranker* `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 470 Mo dans le cache Hugging Face |
| Processus après chargement (graphe, index, BM25, *reranker*, une recherche) | environ 4,7 Gio de mémoire engagée, pointe de 2,8 Gio de mémoire physique |

Les PDF de `data/` ne sont pas nécessaires à l'exécution (le texte des articles vient de l'index). Streamlit ajoute sa
propre mémoire. Sur la machine de mesure (7,8 Gio, dont environ 2,8 Gio libres), le système doit paginer : prévoir au
moins 8 Gio de RAM, davantage pour plusieurs sessions. Les modèles sont partagés entre les sessions d'un même processus.

## Déploiement (non réalisé)

Contraintes relevées :

- **Mémoire** : environ 5 Gio par processus. Les offres gratuites qui donnent environ 1 à 3 Go de RAM ne suffisent pas,
  par exemple Streamlit Community Cloud (limites à vérifier au moment du choix). Il faut une machine ou un conteneur
  d'au moins 8 Gio.
- **Fichiers hors git** : l'index `db_v1_1/` est ignoré par git (`.gitignore`) et doit être copié à part. Les modèles
  doivent être téléchargés une fois, ou copiés dans le cache Hugging Face de la machine cible.
- **Clé** : un hébergement public n'a pas besoin de modifier `agent/` grâce à la clé par session (voir plus haut),
  mais il faut alors choisir :
  - soit chaque visiteur fournit sa clé (`UI_OWNER_KEY=off`) ;
  - soit la clé du propriétaire est partagée avec un plafond quotidien (`UI_OWNER_DAILY_CAP`). La limite gratuite
    actuelle de Groq est d'environ 200 000 *tokens* par jour, soit environ 18 questions par jour en mode Vérifié
    d'après la mesure de latence (environ 10 800 *tokens* par question en moyenne).
- **HTTPS obligatoire** si des visiteurs saisissent leur clé : elle transite par le serveur.
- **Mémoire de conversation** : `MemorySaver` est en RAM et propre au processus. Avec plusieurs processus ou un
  redémarrage, le fil d'une conversation est perdu (la question suivante est traitée seule).
- **Options possibles** : une machine personnelle ou un serveur loué (VM de 8 Gio ou plus) avec `streamlit run` derrière
  un proxy HTTPS ; un conteneur sur un service qui accepte 8 Gio de RAM, par exemple un *Space* Hugging Face CPU (offre
  et limites à vérifier). Aucun déploiement n'a été fait.
