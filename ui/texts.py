"""French interface texts (and the disclaimer in Arabic and English). No logic here."""

TITLE = "Recherche juridique dans les codes tunisiens"

MODE_VERIFIED = "Vérifié"
MODE_FAST = "Rapide"
MODE_HELP = {
    MODE_VERIFIED: ("**Vérifié (v2)** : l'agent analyse la question, peut demander une précision ou refuser une question hors "
                    "champ, cherche les articles, rédige une réponse en affirmations citées, puis vérifie chaque affirmation "
                    "contre l'article cité et retire celles qui ne sont pas confirmées. Plus lent."),
    MODE_FAST: ("**Rapide (v1.1)** : recherche des articles puis réponse en une étape. Plus rapide, mais la réponse n'est "
                "pas vérifiée affirmation par affirmation, et ce mode ne garde pas le fil de la conversation."),
}

DISCLAIMER_FR = ("**Information juridique, pas un conseil juridique.** Les réponses proviennent des textes codifiés indexés et "
                 "peuvent être incomplètes ou erronées. Vérifiez toujours l'article cité, et consultez un professionnel du "
                 "droit pour votre situation.")
DISCLAIMER_OTHER_TITLE = "العربية / English"
DISCLAIMER_AR = ("معلومات قانونية وليست استشارة قانونية. تستند الإجابات إلى النصوص المقننة المفهرسة وقد تكون ناقصة أو خاطئة. "
                 "تحقق دائماً من الفصل المذكور، واستشر مختصاً في القانون بخصوص وضعيتك.")
DISCLAIMER_EN = ("Legal information, not legal advice. Answers come from the indexed codified texts and may be incomplete or "
                 "wrong. Always check the cited article, and consult a legal professional about your situation.")

DEMO_BANNER = ("**Mode démo** : aucune requête n'est envoyée au modèle ; les réponses et les textes d'articles sont des "
               "exemples fictifs, sans valeur juridique.")

NEW_CONVERSATION = "Nouvelle conversation"
INPUT_PLACEHOLDER = "Posez votre question (français, arabe ou anglais)…"
STEPS_TITLE = "Étapes de l'agent"
STEPS_DONE = "Étapes terminées"
STEPS_STOPPED = "Étapes interrompues"
FAST_SPINNER = "Recherche et rédaction en cours…"

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
SOURCES_TITLE = "Articles retrouvés"
UNVERIFIED_BANNER = ("**Réponse non vérifiée** : la vérification affirmation par affirmation a échoué pour cette réponse. "
                     "Lisez les articles cités avant de vous y fier.")
FAST_NOTE = "Mode rapide : cette réponse n'est pas vérifiée affirmation par affirmation."
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

KEY_TITLE = "Clé API"
KEY_LABEL = "Votre clé Groq (facultatif)"
KEY_HELP = ("Utilisée seulement pour vos questions, gardée en mémoire le temps de la session, jamais enregistrée ni "
            "journalisée. Sans clé personnelle, la clé du propriétaire est utilisée si elle est disponible.")
KEY_FORGET = "Oublier ma clé"
KEY_PERSONAL_ACTIVE = "Votre clé personnelle est utilisée."
KEY_OWNER_ACTIVE = "La clé du propriétaire est utilisée."
KEY_OWNER_REMAINING = "Questions restantes aujourd'hui avec la clé du propriétaire : {n}."
KEY_NONE = "Aucune clé disponible : indiquez votre clé Groq pour poser une question."
DEMO_EXAMPLES_TITLE = "Exemples (mode démo)"
