# Guide d'étiquetage des questions nouvelles

Ce guide s'adresse à la personne qui étiquette les questions (connaissances juridiques requises). Votre travail :
pour chaque question posée par une personne réelle, indiquer **les articles des codes tunisiens qui y répondent
vraiment**. Ces étiquettes serviront à mesurer un système de recherche ; elles doivent donc être établies
**sans jamais regarder ce que le système répond**.

## 1. Le fichier

Partez de `fresh/questions_template.json` (copiez-le en `fresh/questions.json`). Chaque entrée de `items`
contient :

| Champ | Contenu |
|---|---|
| `id` | identifiant unique, par exemple `F001`, `F002`… |
| `language` | `fr`, `en` ou `ar` : la langue dans laquelle la personne a écrit (arabe dialectal ou standard : `ar`) |
| `type` | `clear`, `messy`, `vague` ou `out_of_scope` (voir section 3) |
| `question` | la question **exactement telle qu'elle a été tapée** (voir règle 4.4) |
| `gold_articles` | la liste des articles qui répondent, au format décrit en section 2 ; vide pour `out_of_scope` |
| `gold_notes` | remarques libres ; obligatoire pour `vague` (la précision qui lèverait l'ambiguïté) |
| `labeller` | votre nom ou vos initiales |
| `labelled_on` | la date de l'étiquetage, au format `AAAA-MM-JJ` |
| `anonymised` | `true` une fois vérifié qu'il ne reste aucune donnée personnelle (règle 4.5) ; doit être `true` |

Les trois entrées marquées `"example": true` sont des exemples inventés : supprimez-les dans votre fichier
(si vous les gardez, elles sont ignorées).

## 2. Trouver l'article et écrire son identifiant

1. Trouvez l'article dans le texte officiel du code (les PDF du dossier `data/`), comme vous le feriez pour une
   consultation : lisez la section concernée, pas seulement le premier article qui contient le bon mot.
2. Obtenez son identifiant avec la commande (elle lit seulement la liste des articles, pas une réponse du
   système) :

   ```bat
   python fresh\validate_fresh.py --lookup PenalFr 243
   ```

   Elle affiche par exemple `PenalFr/s02/243 | Article 243.- Est puni de cinq ans…`. Vérifiez que le début du
   texte est bien celui de l'article voulu, puis copiez l'identifiant.
3. Format : `<code>/sNN/<article>`, par exemple `PenalFr/s02/243`, `COC/s02/94`. Les articles « bis », « ter »…
   s'écrivent avec un tiret : `PenalFr/s02/240-bis`. Si la commande renvoie plusieurs identifiants pour le même
   numéro (un même numéro dans une loi annexée au code), choisissez celui dont le texte correspond.

Codes couverts (nom à utiliser avec `--lookup`) :

| Code | Nom |
|---|---|
| Code pénal | `PenalFr` |
| Code de procédure pénale | `ProcedurepenaleFr` |
| Code des obligations et des contrats | `COC` |
| Code de procédure civile et commerciale | `Procedurecivilecom` |
| Code de commerce | `CommerceFr` |
| Code des sociétés commerciales | `societeFr` |
| Code du statut personnel | `Statutpersonnel` |
| Code du travail | `travailFr` |
| Code des droits réels | `dtreelFr` |
| Code de la protection de l'enfant | `EnfantFr` |
| Code de la nationalité | `NationaliteFr` |
| Code de droit international privé | `DIPFR` |
| Code de l'arbitrage | `ArbitrageFr` |
| Constitution de 2022 | `tun_RecAcFreeLi` |

## 3. Le type de la question

| Type | Quand |
|---|---|
| `clear` | la question est compréhensible et précise, avec les termes juridiques ou des mots courants corrects |
| `messy` | la question est précise mais mal écrite : fautes, langage familier, abréviations, mauvais terme juridique, ou une situation décrite sans nommer la notion |
| `vague` | il manque une information essentielle : selon la précision, la réponse est dans des articles différents |
| `out_of_scope` | la question ne relève d'aucun des codes ci-dessus (autre pays, autre texte, question non juridique) |

## 4. Règles

1. **Une question par entrée.** Si une personne a écrit plusieurs questions distinctes dans un même message, ne
   le coupez pas vous-même : mettez-le de côté et signalez-le au responsable.
2. **Listez tous les articles qui répondent vraiment** à la question : l'article principal et ceux sans lesquels
   la réponse serait incomplète (une exception, une condition, la peine quand l'article principal y renvoie).
   Ne mettez pas les articles seulement voisins ou « sur le même sujet ».
3. **Questions vagues** : listez les articles qui répondraient selon les lectures plausibles, et écrivez dans
   `gold_notes` la précision qui lèverait l'ambiguïté (par exemple : « s'agit-il d'une affaire civile ou
   pénale ? »).
4. **Ne réécrivez jamais la question** : ni orthographe, ni accents, ni ponctuation, ni traduction. Copiez-la
   telle quelle. Seule exception : la règle 5.
5. **Aucune donnée personnelle** : remplacez tout nom, numéro de téléphone, numéro de carte d'identité, adresse
   ou numéro de dossier par un repère neutre (`[NOM]`, `[NUMÉRO]`, `[ADRESSE]`), sans rien changer d'autre,
   puis mettez `anonymised` à `true`.
6. **Hors champ** : `gold_articles` vide ; indiquez brièvement la raison dans `gold_notes`.
7. **Ne regardez aucune sortie du système** : ne posez pas les questions à l'assistant, n'utilisez pas son
   interface ni ses résultats de recherche, et ne consultez pas les fichiers du dossier `eval/` ni `results/`.
   Ne suggérez pas de sujets ou d'exemples aux personnes qui posent les questions.
8. Si vous hésitez sur un article, écrivez-le dans `gold_notes` plutôt que de l'ajouter à `gold_articles`.

## 5. Vérification

Lancez :

```bat
conda activate rag-app
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
python fresh\validate_fresh.py fresh\questions.json
```

Corrigez les `ERROR` (champ manquant, identifiant inconnu, format de date…). Les `WARNING` de proximité
(« near-duplicate ») signalent une question très proche d'une question déjà utilisée : **ne la modifiez pas**,
transmettez la liste au responsable, qui décidera de l'exclure. Le script indique aussi si les minimums sont
atteints (150 questions, au moins 30 par langue si possible).
