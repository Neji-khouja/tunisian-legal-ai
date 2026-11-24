"""Which law an article belongs to, its date, and whether it was amended or repealed. Written by code from the PDFs, never
by the model. ui/tests/test_ui.py checks that every act named here appears in its PDF."""
import re

# PDF -> (type, number, (day, month, year)) of the act that promulgated or recast the code
ACTS = {
    "ArbitrageFr": ("loi", "93-42", (26, 4, 1993)),
    "COC": ("décret", "", (15, 12, 1906)),
    "CommerceFr": ("loi", "59-129", (5, 10, 1959)),
    "DIPFR": ("loi", "98-97", (27, 11, 1998)),
    "EnfantFr": ("loi", "95-92", (9, 11, 1995)),
    "NationaliteFr": ("décret-loi", "63-6", (28, 2, 1963)),
    "PenalFr": ("décret", "", (9, 7, 1913)),
    "Procedurecivilecom": ("loi", "59-130", (5, 10, 1959)),
    "ProcedurepenaleFr": ("loi", "68-23", (24, 7, 1968)),
    "Statutpersonnel": ("décret", "", (13, 8, 1956)),
    "dtreelFr": ("loi", "65-5", (12, 2, 1965)),
    "societeFr": ("loi", "2000-93", (3, 11, 2000)),
    "travailFr": ("loi", "66-27", (30, 4, 1966)),
    "tun_RecAcFreeLi": ("constitution", "", (25, 7, 2022)),
}

NAMES = {
    "fr": {"ArbitrageFr": "Code de l'arbitrage", "COC": "Code des obligations et des contrats", "CommerceFr": "Code de commerce",
           "DIPFR": "Code de droit international privé", "EnfantFr": "Code de la protection de l'enfant",
           "NationaliteFr": "Code de la nationalité", "PenalFr": "Code pénal",
           "Procedurecivilecom": "Code de procédure civile et commerciale", "ProcedurepenaleFr": "Code de procédure pénale",
           "Statutpersonnel": "Code du statut personnel", "dtreelFr": "Code des droits réels",
           "societeFr": "Code des sociétés commerciales", "travailFr": "Code du travail", "tun_RecAcFreeLi": "Constitution"},
    "en": {"ArbitrageFr": "Arbitration Code", "COC": "Code of Obligations and Contracts", "CommerceFr": "Commercial Code",
           "DIPFR": "Code of Private International Law", "EnfantFr": "Child Protection Code", "NationaliteFr": "Nationality Code",
           "PenalFr": "Penal Code", "Procedurecivilecom": "Code of Civil and Commercial Procedure",
           "ProcedurepenaleFr": "Code of Criminal Procedure", "Statutpersonnel": "Personal Status Code",
           "dtreelFr": "Code of Real Rights", "societeFr": "Commercial Companies Code", "travailFr": "Labour Code",
           "tun_RecAcFreeLi": "Constitution"},
    "ar": {"ArbitrageFr": "مجلة التحكيم", "COC": "مجلة الالتزامات والعقود", "CommerceFr": "المجلة التجارية",
           "DIPFR": "مجلة القانون الدولي الخاص", "EnfantFr": "مجلة حماية الطفل", "NationaliteFr": "مجلة الجنسية التونسية",
           "PenalFr": "المجلة الجزائية", "Procedurecivilecom": "مجلة المرافعات المدنية والتجارية",
           "ProcedurepenaleFr": "مجلة الإجراءات الجزائية", "Statutpersonnel": "مجلة الأحوال الشخصية",
           "dtreelFr": "مجلة الحقوق العينية", "societeFr": "مجلة الشركات التجارية", "travailFr": "مجلة الشغل",
           "tun_RecAcFreeLi": "الدستور"},
}

# sections that are the code itself; the others of a PDF are the promulgating act (s00) or annexed texts
MAIN = {'ArbitrageFr/s01', 'COC/s02', 'CommerceFr/s01', 'DIPFR/s01', 'EnfantFr/s01', 'NationaliteFr/s01', 'PenalFr/s02', 'Procedurecivilecom/s01', 'ProcedurepenaleFr/s01', 'ProcedurepenaleFr/s02', 'Statutpersonnel/s01', 'dtreelFr/s01', 'societeFr/s01', 'travailFr/s01', 'tun_RecAcFreeLi/s00'}

# annexed texts whose section heading names the act
SECTIONS = {
    'ArbitrageFr/s02': 'Loi n° 94-56 du 16 mai 1994',
    'ArbitrageFr/s03': 'Décret n° 95-2084 du 23 octobre 1995',
    'ArbitrageFr/s05': 'Décret n° 2005-1908 du 5 juillet 2005',
    'EnfantFr/s05': 'Décret n° 95-2423 du 11 décembre 1995',
    'EnfantFr/s10': 'Loi n° 98-75 du 28 octobre 1998',
    'EnfantFr/s11': 'Arrêté du Premier ministre du 3 juillet 2002',
    'PenalFr/s01': 'Loi n° 2005-46 du 6 juin 2005',
    'ProcedurepenaleFr/s03': 'Loi n° 2002-94 du 29 octobre 2002',
    'ProcedurepenaleFr/s04': 'Loi n° 2003-75 du 10 décembre 2003',
    'Statutpersonnel/s02': 'Loi n° 57-3 du 1er août 1957',
    'Statutpersonnel/s03': 'Loi n° 2001-31 du 29 mars 2001',
    'Statutpersonnel/s04': 'Arrêté du ministre de la justice du 17 mai 2001',
    'Statutpersonnel/s06': 'Arrêté du chef du gouvernement du 11 janvier 2013',
    'Statutpersonnel/s07': 'Loi n° 98-94 du 9 novembre 1998',
    'Statutpersonnel/s08': 'Loi n° 64-46 du 3 novembre 1964',
    'Statutpersonnel/s11': 'Loi n° 58-27 du 4 mars 1958',
    'Statutpersonnel/s12': 'Décret du 18 juillet 1957',
    'Statutpersonnel/s13': 'Loi n° 98-75 du 28 octobre 1998',
    'Statutpersonnel/s14': 'Loi n° 93-65 du 5 juillet 1993',
    'Statutpersonnel/s15': 'Décret n° 93-1655 du 9 août 1993',
    'dtreelFr/s02': 'Décret-loi n° 64-3 du 20 février 1964',
    'dtreelFr/s03': 'Loi n° 2001-34 du 10 avril 2001',
    'dtreelFr/s06': 'Loi n° 95-21 du 13 février 1995',
    'dtreelFr/s07': 'Décret n° 98-462 du 23 février 1998',
    'societeFr/s02': 'Décret n° 2005-3018 du 21 novembre 2005',
    'societeFr/s03': 'Décret n° 2006-1546 du 6 juin 2006',
    'societeFr/s04': 'Loi n° 95-44 du 2 mai 1995',
    'EnfantFr/s03': "Convention des Nations Unies sur les droits de l'enfant",
    'ArbitrageFr/s04': "Convention relative à l'entraide judiciaire",
}

MONTHS = {
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split(),
    "en": "January February March April May June July August September October November December".split(),
    "ar": "جانفي فيفري مارس أفريل ماي جوان جويلية أوت سبتمبر أكتوبر نوفمبر ديسمبر".split(),
}


_ACT = re.compile(r"(Loi|Décret-loi|Décret) (?:n° ([\d-]+) )?du (\d{1,2})(?:er)? (\w+) (\d{4})")


def date(d: tuple, lang: str = "fr") -> str:
    day, month, year = d
    return f"{'1er' if lang == 'fr' and day == 1 else day} {MONTHS[lang][month - 1]} {year}"


def act(kind: str, number: str, d: tuple, lang: str = "fr") -> str:
    """'loi n° 66-27 du 30 avril 1966' / 'Law No. 66-27 of 30 April 1966' / Arabic."""
    when = date(d, lang)
    if kind == "constitution":
        return when
    if lang == "en":
        name = {"loi": "Law", "décret": "Decree", "décret-loi": "Decree-Law"}[kind]
        return f"{name} No. {number} of {when}" if number else f"{name} of {when}"
    if lang == "ar":
        name = {"loi": "القانون", "décret": "الأمر", "décret-loi": "المرسوم"}[kind]
        return f"{name} عدد {number.split('-')[-1]} لسنة {d[2]} المؤرخ في {when}" if number else f"{name} المؤرخ في {when}"
    return f"{kind} n° {number} du {when}" if number else f"{kind} du {when}"


def law(cid: str, lang: str = "fr") -> tuple[str, str]:
    """(name, date) of the law an article id belongs to; date is "" when the name already carries it or is unknown."""
    stem, section = (cid.split("/") + [""])[:2]
    if stem not in ACTS:
        return stem, ""
    key = f"{stem}/{section}"
    if key in MAIN:
        return NAMES[lang][stem], act(*ACTS[stem], lang)
    if key in SECTIONS:
        m = _ACT.fullmatch(SECTIONS[key])
        if lang == "fr" or not m:
            return SECTIONS[key], ""
        kind, number, d, month, y = m.groups()
        return act(kind.lower(), number or "", (int(d), MONTHS["fr"].index(month) + 1, int(y)), lang), ""
    if section == "s00":                        # the few articles of the act that promulgates the code
        text = act(*ACTS[stem], lang)
        return text[0].upper() + text[1:], ""
    if not re.fullmatch(r"s\d\d", section):
        return NAMES[lang][stem], ""
    return {"fr": "Texte annexé au ", "en": "Text annexed to the ", "ar": "نص ملحق بـ"}[lang] + NAMES[lang][stem], ""


_HEAD = re.compile(r"^\s*(?:Article|Art\.?)\s+(?:premier|\d[\w\s-]{0,16}?)\s*[.\-–:]*\s*\(([^)]{5,220})\)", re.I | re.M)
_NOTE = re.compile(r"(modifié|ajouté|abrogé|complété|remplacé)e?s?\s+(?:par|en vertu)", re.I)


def note(text: str) -> str:
    """What the article says of itself right after its number: 'modifié par la loi n° 89-23 du 27 février 1989', or ''."""
    m = _HEAD.search(text or "")
    if not m or not _NOTE.match(m.group(1).strip()):
        return ""
    s = re.sub(r"\s+", " ", m.group(1)).strip().replace("’", "'")
    s = re.sub(r",\s*(?:portant|relative?|relatif|fixant)\b.*$", "", s)
    s = re.sub(r"n°\s*", "n° ", re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", s)).replace("- du ", " du ")
    s = re.sub(r"^(modifié|ajouté|abrogé|complété|remplacé)e?s?", lambda w: w.group(1).lower(), s, flags=re.I)
    return s


def repealed(text: str) -> bool:
    """The article is marked as repealed, and not replaced since."""
    n = note(text)
    return (n.startswith("abrogé") or " et abrogé" in n) and not re.search(r"abrogé.*\b(?:ajouté|remplacé|modifié)", n)


_NOTE_ACT = re.compile(r"(modifié|ajouté|abrogé|complété|remplacé) par (?:la |le )?(loi|décret-loi|décret) (?:n° ([\d-]+) )?du (\d{1,2})(?:er)? (\w+) (\d{4})")
_VERB = {"en": {"modifié": "amended", "ajouté": "added", "abrogé": "repealed", "complété": "supplemented", "remplacé": "replaced"},
         "ar": {"modifié": "نُقّح", "ajouté": "أُضيف", "abrogé": "أُلغي", "complété": "تُمّم", "remplacé": "عُوّض"}}


def note_in(text: str, lang: str) -> str:
    """note(), in the language of the answer; '' when it cannot be put in that language exactly."""
    n = note(text)
    if lang == "fr" or not n:
        return n
    m = _NOTE_ACT.fullmatch(n)
    if not m or m.group(5) not in MONTHS["fr"]:
        return ""
    verb, kind, number, d, month, y = m.groups()
    named = act(kind, number or "", (int(d), MONTHS["fr"].index(month) + 1, int(y)), lang)
    return f"{_VERB[lang][verb]} by {named}" if lang == "en" else f"{_VERB[lang][verb]} بمقتضى {named}"


def _of(name: str) -> str:
    """French 'of': du Code pénal, de la loi n° 98-94..., de l'arrêté..."""
    low = name[0].lower() + name[1:]
    if name.startswith(("Constitution", "Convention", "Loi")):
        return "de la " + (name if name.startswith("Constitution") else low)
    if name.startswith("Arrêté"):
        return "de l'" + low
    return "du " + (name if name.startswith("Code") else low)


def _list(items: list, word: str) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + (" و" if word == "و" else f" {word} ") + items[-1]


def sentence(cids: list[str], texts: dict, lang: str = "fr") -> str:
    """One sentence saying where the rules of an answer come from: the articles, their law and its date; when a single
    article is cited, also its amendment. texts: article id -> article text."""
    lang = lang if lang in NAMES else "fr"
    groups: dict = {}
    for cid in cids:
        groups.setdefault(law(cid, lang), []).append(cid)
    if not groups:
        return ""
    many = len(cids) > 1
    parts = []
    for (name, when), ids in groups.items():
        nums = [i.split("/")[-1] for i in ids][:6]
        tail = f" ({when})" if when else ""
        if lang == "en":
            parts.append(f"article{'s' if len(ids) > 1 else ''} {_list(nums, 'and')} of {'' if name.startswith(('Law', 'Decree')) else 'the '}{name}{tail}")
        elif lang == "ar":
            parts.append(f"{'الفصول' if len(ids) > 1 else 'الفصل'} {_list(nums, 'و')} من {name}{tail}")
        else:
            parts.append(f"{'aux articles' if len(ids) > 1 else chr(224) + chr(32) + 'l' + chr(39) + 'article'} {_list(nums, 'et')} {_of(name)}{tail}")
    if lang == "en":
        out = f"{'These rules are' if many else 'This rule is'} in {_list(parts, 'and')}."
    elif lang == "ar":
        out = f"{'وردت هذه القواعد في' if many else 'وردت هذه القاعدة في'} {_list(parts, 'و')}."
    else:
        out = f"{'Ces règles figurent' if many else 'Cette règle figure'} {_list(parts, 'et')}."
    if not many:
        n = note_in(texts.get(cids[0], ""), lang)
        if n and not repealed(texts.get(cids[0], "")):
            out += {"fr": f" Cet article a été {n}.", "en": f" This article was {n}.", "ar": f" {n}."}[lang]
    return out
