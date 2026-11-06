"""English text frontend: context-aware (POS) G2P producing ARPAbet phone strings.

Differences from the upstream FastSpeech2 `preprocess_english`:
  * pronunciations come from g2p_en with POS-based homograph handling plus a small
    noun/verb heteronym table (e.g. CONduct vs conDUCT), instead of a context-blind
    lexicon lookup;
  * the same function builds the MFA dictionary (all CMUdict variants) so alignment
    and synthesis use one phone inventory.
"""
import re
import sys

from text.cleaners import english_cleaners

_G2P = None
_TOKENIZER = None

# noun/adjective -> first, verb -> second.  All symbols are valid CMUdict ARPAbet.
HETERONYMS = {
    "conduct": ("K AA1 N D AH0 K T", "K AH0 N D AH1 K T"),
    "contract": ("K AA1 N T R AE2 K T", "K AH0 N T R AE1 K T"),
    "object": ("AA1 B JH EH0 K T", "AH0 B JH EH1 K T"),
    "permit": ("P ER1 M IH2 T", "P ER0 M IH1 T"),
    "present": ("P R EH1 Z AH0 N T", "P R IY0 Z EH1 N T"),
    "project": ("P R AA1 JH EH0 K T", "P R AH0 JH EH1 K T"),
    "record": ("R EH1 K ER0 D", "R IH0 K AO1 R D"),
    "refuse": ("R EH1 F Y UW2 Z", "R IH0 F Y UW1 Z"),
    "subject": ("S AH1 B JH IH0 K T", "S AH0 B JH EH1 K T"),
    "rebel": ("R EH1 B AH0 L", "R IH0 B EH1 L"),
    "increase": ("IH1 N K R IY2 S", "IH0 N K R IY1 S"),
    "progress": ("P R AA1 G R EH2 S", "P R AH0 G R EH1 S"),
    "insult": ("IH1 N S AH0 L T", "IH0 N S AH1 L T"),
    "produce": ("P R OW1 D UW0 S", "P R AH0 D UW1 S"),
    "suspect": ("S AH1 S P EH0 K T", "S AH0 S P EH1 K T"),
    "convict": ("K AA1 N V IH0 K T", "K AH0 N V IH1 K T"),
}
_PUNCT = set(",;.:?!-")


def _g2p():
    global _G2P
    if _G2P is None:
        from g2p_en import G2p

        _G2P = G2p()
    return _G2P


def _tokenize(text):
    global _TOKENIZER
    if _TOKENIZER is None:
        from nltk.tokenize import TweetTokenizer

        _TOKENIZER = TweetTokenizer()
    return _TOKENIZER.tokenize(text)


def _pos_tags(tokens):
    try:
        from nltk import pos_tag

        return [t for _, t in pos_tag(tokens)]
    except LookupError:
        print(
            "WARNING: NLTK POS tagger data missing - heteronyms will not be disambiguated. "
            "Run: python -c \"import nltk; nltk.download('averaged_perceptron_tagger'); "
            "nltk.download('averaged_perceptron_tagger_eng')\"",
            file=sys.stderr,
        )
        return ["NN"] * len(tokens)


def english_words(text):
    """Cleaned words (lower-case, apostrophes kept, no punctuation) for MFA .lab files."""
    t = english_cleaners(text)
    t = re.sub(r"[^a-z' ]", " ", t)
    t = re.sub(r"\s+'|'\s+", " ", t)
    return [w for w in t.split() if w and re.search("[a-z]", w)]


def word_pronunciations(word):
    """All pronunciation variants (list of phone lists) for dictionary building."""
    g = _g2p()
    w = word.lower()
    prons = []
    if w in HETERONYMS:
        prons += [p.split() for p in HETERONYMS[w]]
    if w in g.cmu:
        for p in g.cmu[w]:
            if p not in prons:
                prons.append(p)
    if not prons:
        prons.append(g.predict(w))
    return prons


def english_to_phone_string(text, g2p=None, tagger=None):
    """Raw English text -> '{HH AH0 L OW1 sp ...}' (trailing punctuation dropped)."""
    g = g2p or _g2p()
    cleaned = english_cleaners(text)
    cleaned = re.sub(r"[^a-z' ,;.:?!\-]", " ", cleaned)
    tokens = _tokenize(cleaned)
    tags = (tagger or _pos_tags)(tokens)

    phones = []
    for tok, tag in zip(tokens, tags):
        if re.search("[a-z]", tok) is None:
            if tok in _PUNCT or any(c in _PUNCT for c in tok):
                phones.append("sp")
            continue
        if tok in HETERONYMS:
            pron = HETERONYMS[tok][1 if tag.startswith("VB") else 0].split()
        elif tok in getattr(g, "homograph2features", {}):
            pron1, pron2, pos1 = g.homograph2features[tok]
            pron = pron1 if tag.startswith(pos1) else pron2
        elif tok in g.cmu:
            pron = g.cmu[tok][0]
        else:
            pron = g.predict(tok)
        phones.extend(pron)
    while phones and phones[-1] == "sp":
        phones.pop()
    return "{" + " ".join(phones) + "}"
