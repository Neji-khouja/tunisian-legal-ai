"""French interface texts (and the disclaimer in Arabic and English). No logic here."""

TITLE = "Codes tunisiens"
SUBTITLE = "Posez une question de droit en français, arabe ou anglais. Les articles utilisés sont affichés sous la réponse."

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
    "answer": "Lecture des articles et réponse",
    "not_found": "Aucun article ne répond",
    "verify": "Vérification de chaque affirmation",
    "repair": "Correction de la réponse",
    "finalize": "Finalisation",
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

MODE_CHOICES = [("Agent", "agent"), ("Rapide", "fast")]
MODE_NAMES = {"agent": "agent", "fast": "rapide"}
MODE_HELP = {
    "agent": ("Plusieurs étapes : comprend la conversation, demande une précision si besoin, refuse ce qui sort des codes "
              "et vérifie chaque affirmation contre son article. Plus lent."),
    "fast": ("Une seule recherche et une seule réponse. Rapide et économe, mais sans mémoire de la conversation et sans "
             "vérification."),
}
WORDS = {"fr": {"sources": "Sources", "note": "Remarque", "articles": "Articles applicables", "others": "Autres articles consultés ({n})"},
         "en": {"sources": "Sources", "note": "Note", "articles": "Applicable articles", "others": "Other articles consulted ({n})"},
         "ar": {"sources": "المصادر", "note": "ملاحظة", "articles": "الفصول المنطبقة", "others": "فصول أخرى ({n})"}}
BRAND_SUB = "Assistant juridique"
MODE_LABEL = "Mode"
HISTORY_LABEL = "Conversations"
HISTORY_EMPTY = "Aucune conversation pour l'instant."
NEW_CHAT = "Nouvelle conversation"
DELETE_CHAT = "Supprimer la conversation"
SEND = "Envoyer"
TOKENS_LINE = "{total} tokens (entrée {tin}, sortie {tout}) · {sec} s · mode {mode}"
TOKENS_CACHED = ", dont {n} en cache"
TOKENS_REASONING = ", dont {n} de réflexion"
TOKENS_HIT = "En cache (hors limites)"
FROM_CACHE = "Réponse gardée en mémoire : cette question a déjà été posée. Le modèle n'a pas été appelé (0 token)."
QUOTA_TITLE = "Tokens restants (24 h)"
QUOTA_OWNER = "Clé utilisée : celle de l'application"
QUOTA_PERSONAL = "Clé utilisée : la vôtre"
QUOTA_NONE = "Aucune clé. Seules les réponses déjà gardées en mémoire sont possibles ; indiquez votre clé Groq ci-dessous."
QUOTA_ABOUT = "Environ {agent} questions Agent ou {fast} Rapide"
QUOTA_QUESTIONS = "Questions autorisées aujourd'hui : {n}"
QUOTA_HINT = "Estimation d'après l'usage ici sur 24 heures."
REPEALED = "Attention : {label} est indiqué comme {note}. Vérifiez le texte en vigueur."
TOKENS_TITLE = "Tokens utilisés"
TOKENS_CONV = "Cette conversation"
TOKENS_DEVICE = "Toutes les conversations"
TAB_CHAT = "Discussion"
TAB_HOW = "Comment ça marche"
ABOUT_INTRO = """## Comment ça marche

Cet assistant répond à des questions sur 14 textes juridiques tunisiens : Constitution, code pénal, code du travail, code des obligations et des contrats, statut personnel, commerce, sociétés, procédure civile et pénale, etc. Il répond à partir de ces textes et affiche sous chaque réponse les articles utilisés ; dépliez un article pour lire son texte.

Aucun des deux modes n'est infaillible : l'article qui répond à la question peut ne pas être trouvé. Lisez toujours l'article cité."""
ABOUT_STORY = """### En image : le bibliothécaire et le juriste

Imaginez une bibliothèque de droit.

- **Le bibliothécaire** (la recherche) va chercher les articles les plus proches de votre question. Il ne répond pas.
- **Le juriste** (le modèle) lit ces articles et vous répond.
- **Le relecteur** (la vérification, en mode Agent seulement) compare chaque phrase du juriste à l'article cité.

**En mode Rapide**, le bibliothécaire apporte 6 articles et le juriste répond tout de suite. Personne ne relit.

**En mode Agent**, le bibliothécaire apporte 8 articles, et le juriste se demande d'abord s'ils répondent vraiment :

- *Ils répondent.* Il rédige, puis le relecteur contrôle chaque phrase.
- *Ils ne répondent pas.* Il renvoie le bibliothécaire avec une demande reformulée, une seule fois.
- *La question peut vouloir dire plusieurs choses.* Il vous demande une précision.
- *Toujours rien.* Il vous le dit, sans inventer.

**Deux exemples en mode Agent**

- « Quelle est la peine pour le vol simple ? » Parmi les articles apportés, l'article 264 du Code pénal répond : le juriste rédige et le relecteur contrôle.
- « Quelle est la peine pour un excès de vitesse ? » Le code de la route ne fait pas partie de la bibliothèque : le juriste renvoie une fois le bibliothécaire, puis vous dit qu'il n'a rien trouvé."""
ABOUT_MODES = """### Les deux modes

| | **Agent** | **Rapide** |
|---|---|---|
| Principe | plusieurs étapes enchaînées | une recherche, puis une réponse |
| Recherche | question reformulée en vocabulaire juridique, priorité au code concerné, articles liés ajoutés, nouvelles recherches si besoin | question telle quelle (traduite depuis l'arabe ou l'anglais) |
| Question vague | demande une précision | répond quand même |
| Hors des codes | refuse | essaie de répondre |
| Mémoire | tient compte des messages précédents | chaque question est traitée seule |
| Vérification | chaque affirmation est contrôlée contre son article | aucune |
| Coût | 4 appels au modèle ou plus, environ 8 000 à 10 000 tokens par question | 1 ou 2 appels, environ 2 000 tokens |
| À choisir pour | une question de suivi, une question vague, une réponse dont chaque phrase est contrôlée | une question précise sur une règle ou un article |

L'Agent consomme plus parce qu'il fait plus de choses : chaque étape est un appel au modèle, et plusieurs relisent les articles. Les deux modes cherchent dans les mêmes textes avec le même moteur : si l'article n'est pas trouvé, aucun des deux ne peut répondre."""
TAB_AGENT = "Mode Agent"
TAB_FAST = "Mode Rapide"
ABOUT_AGENT = """### Le parcours de l'agent

L'agent enchaîne des étapes et peut revenir en arrière. Chaque encadré bleu ou vert est un appel au modèle.

- **condense** : réécrit une question de suivi pour qu'elle se comprenne seule (à partir du deuxième message).
- **analyze** : détecte la langue, l'intention (répondre, préciser, refuser) et les codes concernés.
- **normalize** : traduit la question en français si besoin, puis la reformule en vocabulaire juridique.
- **retrieve** : lance la recherche du mode Rapide avec la question d'origine et la question reformulée, favorise le code concerné, garde 8 articles et ajoute ceux auxquels ils renvoient.
- **answer** : lit les articles et juge s'ils répondent. Si oui, écrit une courte explication, chaque phrase avec les articles qui la fondent. Si la question est peu claire au vu des articles, vous demande une précision (**clarify**).
- **reformulate / retrieve_more** : si les articles ne répondent pas, réécrit la recherche, cherche encore une fois et revient à **answer**. Si rien ne répond, l'agent le dit (**not_found**) au lieu d'inventer.
- **détails** : réponse courte par défaut. Demandez « plus de détails » pour une réponse plus complète.
- **verify / repair** : contrôle chaque phrase contre son article, puis corrige ou retire seulement celles qui ne sont pas confirmées.

La date de la loi et les mentions « modifié » ou « abrogé » ne viennent pas du modèle : elles sont lues dans les textes."""
ABOUT_FAST = """### Le parcours du mode Rapide

Une seule passe, sans retour en arrière : chercher, classer, répondre. La recherche se fait sur le serveur et ne consomme aucun token.

1. **Langue** : une question en français est cherchée telle quelle. Une question en arabe ou en anglais est d'abord traduite (un appel au modèle), et cherchée aussi dans sa langue d'origine.
2. **Trois recherches** : par le sens (chaque article est un vecteur ; on prend les 20 plus proches), par mots-clés (BM25, 15 articles) et par numéro d'article si vous en citez un.
3. **Fusion** : les listes sont réunies ; un article bien classé dans plusieurs listes remonte.
4. **Reranking** : un second modèle, plus précis mais plus lent, relit chaque candidat avec la question et garde les 6 meilleurs.
5. **Réponse** : le modèle lit ces 6 articles et répond en un seul appel.

Ce que ce mode ne fait pas : il ne se souvient pas de la conversation, ne demande pas de précision, ne relance pas la recherche et ne contrôle pas sa réponse."""
ABOUT_TOKENS = """### Tokens

Un token est un morceau de mot que le modèle lit ou écrit. Le compteur additionne tous les appels au modèle faits pour une question : l'entrée est ce que le modèle lit (la question et les articles), la sortie ce qu'il écrit, y compris sa réflexion, qui n'est pas affichée. Les tokens en cache sont des débuts de texte identiques d'un appel à l'autre : le fournisseur ne les compte pas dans ses limites. Une question déjà posée telle quelle par quelqu'un reçoit la réponse gardée en mémoire, sans nouveau token : c'est alors écrit au-dessus de la réponse.

Les tokens restants sont une estimation : le fournisseur ne publie pas ce chiffre. L'application retire de la limite quotidienne les tokens utilisés ici depuis 24 heures ; l'usage de la même clé ailleurs n'est pas compté.

Si vous n'indiquez pas de clé, l'application utilise la sienne, avec un nombre limité de questions par jour. Sans aucune clé, elle ne répond pas à une question nouvelle."""
ABOUT_DATA = """### Vos données

Les questions et les réponses sont enregistrées sur le serveur pour évaluer le système. La liste des conversations, elle, est gardée dans votre navigateur ; la supprimer n'efface pas cet enregistrement."""
