"""Shared, dependency-free text utilities: bilingual (DE/EN) normalisation, tokenisation, number extraction."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüßÀ-ÿğĞışŞİçÇ0-9]+(?:[.,][0-9]+)?", re.UNICODE)
NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w])")

STOPWORDS = {
    # English
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "by", "with", "at",
    "as", "it", "this", "that", "can", "i", "my", "me", "do", "does", "how", "what", "when", "which",
    "who", "will", "would", "should", "may", "per", "from", "if", "any", "our", "we", "you", "your",
    "there", "their", "they", "have", "has", "was", "were", "about", "into", "than", "also", "so",
    "according", "based", "policy", "after", "before", "must", "all", "not", "no", "up", "its",
    # German
    "der", "die", "das", "und", "oder", "ein", "eine", "einen", "ist", "sind", "zu", "im", "in", "für",
    "mit", "von", "auf", "bei", "wie", "was", "wann", "ich", "mein", "meine", "es", "den", "dem", "des",
    "nach", "vor", "kann", "werden", "wird", "nicht", "auch", "sich",
}

# Bilingual concept canonicalisation: maps surface stems to a shared concept token.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "probation": ("probation", "probezeit", "trial period", "probationary"),
    "vacation": ("vacation", "urlaub", "holiday", "holidays", "annual leave", "leave days", "urlaubstage", "pto"),
    "salary": ("salary", "gehalt", "pay", "compensation", "vergütung", "verguetung", "wage", "band", "bands", "entgelt"),
    "notice": ("notice", "kündigung", "kuendigung", "kündigungsfrist", "resign", "resignation", "termination", "terminate"),
    "parental": ("parental", "elternzeit", "maternity", "paternity", "mutterschutz", "elterngeld", "baby", "child"),
    "remote": ("remote", "home office", "homeoffice", "mobile arbeit", "work from home", "wfh", "telework"),
    "privacy": ("privacy", "gdpr", "dsgvo", "datenschutz", "personal data", "retention", "applicant data", "erasure"),
    "offer": ("offer", "angebot", "final offer", "counteroffer", "counter-offer", "negotiate", "negotiation"),
    "discrimination": ("discrimination", "diskriminierung", "agg", "harassment", "belästigung", "complaint", "beschwerde"),
    "sick": ("sick", "krank", "krankmeldung", "illness", "sick note"),
}

_CONCEPT_LOOKUP: list[tuple[str, str]] = sorted(
    ((surface, concept) for concept, surfaces in CONCEPTS.items() for surface in surfaces),
    key=lambda x: -len(x[0]),
)

_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "ae", "Ö": "oe", "Ü": "ue"})
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})


def stable_unit(*parts: object) -> float:
    """Deterministic pseudo-random number in [0, 1) derived from the given parts (no global RNG state)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def stable_int(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


# Small German -> English stem lexicon so bilingual queries match the (English) policy corpus.
_DE_EN = {
    "verlaengert": "extend", "verlaenger": "extend", "tag": "day", "tage": "day", "woche": "week", "woch": "week",
    "monat": "month", "monate": "month", "jahr": "year", "jahre": "year", "darf": "may", "arbeit": "work",
    "arbeiten": "work", "gespeichert": "stor", "beantragt": "request", "beantrag": "request", "angestellt": "employe",
    "mitarbeit": "employe", "bewerb": "applicant", "frist": "period",
}


def stem(token: str) -> str:
    token = token.translate(_UMLAUTS)
    if token in _DE_EN:
        return _DE_EN[token]
    for suffix in ("ungen", "ung", "ern", "en", "es", "er", "ed", "e", "s"):
        if len(token) > 5 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    return _DE_EN.get(token, token)


def concepts_in(text: str) -> set[str]:
    low = (text or "").lower()
    return {concept for surface, concept in _CONCEPT_LOOKUP if surface in low}


def content_tokens(text: str) -> list[str]:
    """Stopword-filtered stems plus canonical concept tokens (enables DE/EN matching)."""
    toks = [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]
    toks.extend(f"concept:{c}" for c in sorted(concepts_in(text)))
    return toks


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in NUMBER_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", ".")))
        except ValueError:
            continue
    return out


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ\"(\[])", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def leet_normalize(text: str) -> str:
    """Undo common leetspeak substitutions inside alphabetic words (e.g. 'ign0r3' -> 'ignore')."""

    def fix(m: re.Match[str]) -> str:
        w = m.group(0)
        return w.translate(_LEET) if re.search(r"[A-Za-z]", w) else w

    return re.sub(r"[A-Za-z0-9@$!]{3,}", fix, text)


def decode_base64_segments(text: str) -> list[str]:
    """Return printable decodings of base64-looking segments (used by both attacker-sim and defender)."""
    decoded = []
    for seg in re.findall(r"[A-Za-z0-9+/]{16,}={0,2}", text or ""):
        try:
            raw = base64.b64decode(seg + "=" * (-len(seg) % 4), validate=True)
            s = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if s and sum(ch.isprintable() for ch in s) / len(s) > 0.95 and re.search(r"[a-zA-Z]{3,}", s):
            decoded.append(s)
    return decoded


def join_split_payloads(text: str) -> list[str]:
    """Reassemble payload-splitting attacks such as: a="ignore all"; b="rules"; run a+b."""
    assigns = dict(re.findall(r"\b([a-zA-Z])\s*=\s*[\"']([^\"']+)[\"']", text or ""))
    joined = []
    for combo in re.findall(r"\b([a-zA-Z](?:\s*\+\s*[a-zA-Z])+)\b", text or ""):
        names = [n.strip() for n in combo.split("+")]
        if all(n in assigns for n in names):
            joined.append(" ".join(assigns[n] for n in names))
    return joined


def expanded_views(text: str) -> list[str]:
    """All textual views an instruction-following model (or a defender) may 'see' in the input."""
    views = [text, leet_normalize(text)]
    views.extend(decode_base64_segments(text))
    views.extend(join_split_payloads(text))
    return views


def repetition_ratio(text: str) -> float:
    """1 - unique/total tokens: high values indicate token flooding."""
    toks = tokenize(text)
    if not toks:
        return 0.0
    return 1.0 - len(set(toks)) / len(toks)
