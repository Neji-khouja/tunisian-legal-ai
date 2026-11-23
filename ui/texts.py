"""French interface texts (and the disclaimer in Arabic and English). No logic here."""

TITLE = "Codes tunisiens"
SUBTITLE = "Posez une question de droit en français, arabe ou anglais. Chaque affirmation cite l'article dont elle vient."

DISCLAIMER_FR = ("**Information juridique, pas un conseil juridique.** Les réponses proviennent des textes codifiés indexés et "
                 "peuvent être incomplètes ou erronées. Vérifiez toujours l'article cité, et consultez un professionnel du "
                 "droit pour votre situation.")
DISCLAIMER_OTHER_TITLE = "العربية / English"
DISCLAIMER_AR = ("معلومات قانونية وليست استشارة قانونية. تستند الإجابات إلى النصوص المقننة المفهرسة وقد تكون ناقصة أو خاطئة. "
                 "تحقق دائماً من الفصل المذكور، واستشر مختصاً في القانون بخصوص وضعيتك.")
DISCLAIMER_EN = ("Legal information, not legal advice. Answers come from the indexed codified texts and may be incomplete or "
                 "wrong. Always check the cited article, and consult a legal professional about your situation.")

DEMO_BANNER = "Mode démo : réponses fictives, aucun modèle appelé."

LOG_NOTICE = ("Les questions et les réponses sont enregistrées pour évaluer le système. N'indiquez pas de données "
              "personnelles. Votre clé API n'est jamais enregistrée.")

NEW_CONVERSATION = "Nouvelle conversation"
INPUT_PLACEHOLDER = "Posez votre question (français, arabe ou anglais)…"
STEPS_TITLE = "Étapes de l'agent"
STEPS_DONE = "Étapes terminées"
STEPS_STOPPED = "Étapes interrompues"

NODE_LABELS = {
    "condense": "Prise en compte de la conversation",
    "analyze": "Analyse de la question",
    "clarify": "Demande de précision",
    "refuse": "Question hors du champ couvert",
    "retrieve": "Recherche des articles",
    "grade": "Évaluation des articles trouvés",
    "reformulate": "Reformulation de la recherche",
    "retrieve_more": "Recherche complémentaire",
    "answer": "Rédaction de la réponse",
    "verify": "Vérification de chaque affirmation",
    "repair": "Correction de la réponse",
    "finalize": "Finalisation",
}

CODE_NAMES = {
    "ArbitrageFr": "Code de l'arbitrage",
    "COC": "Code des obligations et des contrats",
    "CommerceFr": "Code de commerce",
    "DIPFR": "Code de droit international privé",
    "EnfantFr": "Code de la protection de l'enfant",
    "NationaliteFr": "Code de la nationalité",
    "PenalFr": "Code pénal",
    "Procedurecivilecom": "Code de procédure civile et commerciale",
    "ProcedurepenaleFr": "Code de procédure pénale",
    "Statutpersonnel": "Code du statut personnel",
    "dtreelFr": "Code des droits réels",
    "societeFr": "Code des sociétés commerciales",
    "travailFr": "Code du travail",
    "tun_RecAcFreeLi": "Constitution",
}

ARTICLE = "art."
CITATION_UNAVAILABLE = "Texte de l'article indisponible."
UNVERIFIED_BANNER = ("**Réponse non vérifiée** : la vérification affirmation par affirmation a échoué pour cette réponse. "
                     "Lisez les articles cités avant de vous y fier.")
DROPPED = "{n} affirmation(s) retirée(s), car les articles cités ne les confirment pas."
CAVEAT = "Remarque : {text}"
REWRITTEN = "Question comprise comme : {text}"
CLARIFY_HINT = "Répondez dans la zone de saisie pour préciser votre question."
FALLBACK_NOTES = {
    "analyze_failed": "L'analyse de la question a échoué ; l'agent a continué sans elle.",
    "condense_failed": "Le lien avec les messages précédents n'a pas pu être fait ; la question a été traitée seule.",
}
RATE_LIMIT_IN_STEP = "Une étape a rencontré la limite de requêtes du fournisseur du modèle."

ERRORS = {
    "quota_daily": ("La limite quotidienne de requêtes du modèle est atteinte. Réessayez plus tard (souvent le lendemain), "
                    "ou indiquez votre propre clé dans le panneau latéral."),
    "rate_limit": "Trop de requêtes en peu de temps. Patientez une minute, puis reposez la question.",
    "auth": "La clé API a été refusée. Vérifiez la clé indiquée dans le panneau latéral.",
    "network": "Le service du modèle est momentanément injoignable. Vérifiez la connexion, puis reposez la question.",
    "no_key": ("Aucune clé API n'est disponible pour cette question. Indiquez votre propre clé Groq dans le panneau "
               "latéral."),
    "owner_cap": ("Le nombre de questions autorisé aujourd'hui avec la clé du propriétaire est atteint. Indiquez votre propre "
                  "clé Groq dans le panneau latéral, ou réessayez demain."),
    "other": "Une erreur est survenue. Réessayez ; si elle se répète, signalez-la au propriétaire de l'application.",
}

KEY_LABEL = "Votre clé Groq (facultatif)"
KEY_HELP = "Gardée en mémoire le temps de la session, jamais enregistrée."
KEY_FORGET = "Oublier ma clé"
KEY_PERSONAL_ACTIVE = "Votre clé est utilisée."
KEY_OWNER_REMAINING = "Questions restantes aujourd'hui : {n}."
KEY_NONE = "Aucune clé disponible : indiquez votre clé Groq pour poser une question."

NODE_LABELS["retrieve"] = "Recherche des articles"
MODE_CHOICES = [("Agent (vérifie chaque affirmation)", "agent"), ("Rapide (une seule passe)", "fast")]
MODE_NAMES = {"agent": "agent", "fast": "rapide"}
MODE_HELP = {
    "agent": ("**Agent** : comprend la conversation, demande une précision si la question est vague, refuse ce qui sort des "
              "codes, cherche, juge les articles trouvés, rédige puis **vérifie chaque affirmation** contre l'article cité. "
              "Plus lent et plus de tokens."),
    "fast": ("**Rapide** : reformule la question, cherche et classe les articles, puis le modèle répond **une seule fois**. "
             "Peu de tokens, mais sans mémoire de la conversation et sans vérification : lisez bien les articles cités."),
}
SOURCES_TITLE = "Articles fournis au modèle"
HISTORY_LABEL = "Conversations"
NEW_CHAT = "＋ Nouvelle conversation"
DELETE_CHAT = "Supprimer cette conversation"
TOKENS_LINE = "{total} tokens (entrée {tin}, sortie {tout}) · {sec} s · mode {mode}"
TOKENS_METER = "**Tokens** · cette conversation : {conv} · cet appareil : {device}"
TAB_CHAT = "Discussion"
TAB_HOW = "Comment ça marche"
ABOUT_INTRO = (
    "## Comment ça marche\n\n"
    "Cet assistant répond à des questions sur 14 textes juridiques tunisiens (code pénal, code du travail, code des "
    "obligations et des contrats, statut personnel, commerce, sociétés, procédure civile et pénale, etc.). Il ne répond "
    "qu'à partir de ces textes : chaque affirmation cite l'article dont elle vient, et vous pouvez lire cet article en "
    "dépliant la citation.")
ABOUT_MODES = """### Les deux modes

| | **Agent** | **Rapide** |
|---|---|---|
| Principe | plusieurs étapes enchaînées, chacune pouvant corriger la précédente | une seule passe : chercher puis répondre |
| Question vague | demande une précision | répond quand même |
| Hors des codes | refuse poliment | essaie de répondre |
| Conversation | se souvient des messages précédents | chaque question est seule |
| Vérification | chaque affirmation est contrôlée contre son article ; ce qui n'est pas confirmé est retiré | aucune |
| Coût | plusieurs appels au modèle : plus de tokens, plus lent | un ou deux appels : peu de tokens |

Les deux modes utilisent la **même recherche** (ci-dessous) ; seule la suite change."""
ABOUT_RETRIEVAL = """### La recherche d'articles

1. La question est réécrite en français juridique, pour retrouver le texte même si vous écrivez en arabe, en anglais ou avec des fautes.
2. Deux recherches en parallèle : par le sens (sémantique) et par mots-clés (BM25). Si vous citez « article 243 », cet article est lu directement.
3. Les résultats sont fusionnés puis classés par un second modèle ; les 6 meilleurs articles sont gardés.
4. Les articles auxquels ils renvoient sont ajoutés."""
ABOUT_AGENT = """### Le parcours de l'agent

- **condense** : réécrit une question de suivi pour qu'elle se comprenne seule.
- **analyze** : détecte la langue, l'intention (répondre, préciser, refuser) et les codes concernés.
- **retrieve** : la recherche ci-dessus.
- **grade / reformulate** : vérifie que les articles suffisent, sinon reformule la recherche (deux fois au plus).
- **answer** : écrit des affirmations courtes, chacune avec les articles qui la fondent.
- **verify / repair** : contrôle chaque affirmation contre son article ; corrige ou retire ce qui n'est pas confirmé."""
ABOUT_TOKENS = """### Tokens

Un token est un morceau de mot que le modèle lit ou écrit. Le compteur additionne tous les appels au modèle faits pour une question : l'entrée est ce que le modèle lit (la question et les articles), la sortie ce qu'il écrit. Le fournisseur impose une limite quotidienne ; le mode Rapide en consomme beaucoup moins."""
ABOUT_DATA = """### Vos données

Les questions et les réponses sont enregistrées pour évaluer le système. Les conversations affichées à gauche ne sont gardées que dans votre navigateur."""
