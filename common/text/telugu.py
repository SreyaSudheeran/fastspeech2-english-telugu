"""Telugu text frontend: normalisation, number expansion and rule-based G2P.

Telugu orthography is almost phonemic, so a deterministic script-to-phone map is
used instead of a trained G2P model. The same function is used to
  (a) write the pronunciation dictionary handed to the Montreal Forced Aligner, and
  (b) convert test sentences at inference time,
which keeps training and inference phone sequences consistent.
"""
import re
import unicodedata

# ----------------------------------------------------------------------------
# Phone inventory (lower-case ASCII only; MFA-safe, no digits)
# ----------------------------------------------------------------------------
TELUGU_VOWELS = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo", "ai", "au"]
TELUGU_CONSONANTS = [
    "k", "kh", "g", "gh", "ng",
    "c", "ch", "j", "jh", "ny",
    "tx", "txh", "dx", "dxh", "nx",      # retroflex
    "t", "th", "d", "dh", "n",           # dental
    "p", "ph", "b", "bh", "m",
    "y", "r", "l", "lx", "v",
    "sh", "sx", "s", "h",
]
TELUGU_PHONES = TELUGU_VOWELS + TELUGU_CONSONANTS

_INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"],
    "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}
_VOWEL_SIGNS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"],
    "ృ": ["r", "u"], "ౄ": ["r", "uu"], "ె": ["e"], "ే": ["ee"], "ై": ["ai"],
    "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
}
_CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tx", "ఠ": "txh", "డ": "dx", "ఢ": "dxh", "ణ": "nx",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "lx", "వ": "v",
    "శ": "sh", "ష": "sx", "స": "s", "హ": "h",
}
_VIRAMA = "్"
_ANUSVARA = {"ం", "ఁ"}
_VISARGA = "ః"
_IGNORED = {"‌", "‍"}  # ZWNJ / ZWJ

# Homorganic nasal for anusvara, decided by the following consonant.
_NASAL_BEFORE = {}
for _c in ["k", "kh", "g", "gh"]:
    _NASAL_BEFORE[_c] = "ng"
for _c in ["c", "ch", "j", "jh"]:
    _NASAL_BEFORE[_c] = "ny"
for _c in ["tx", "txh", "dx", "dxh"]:
    _NASAL_BEFORE[_c] = "nx"
for _c in ["t", "th", "d", "dh"]:
    _NASAL_BEFORE[_c] = "n"
for _c in ["p", "ph", "b", "bh"]:
    _NASAL_BEFORE[_c] = "m"

# ----------------------------------------------------------------------------
# Number expansion (Telugu words)
# ----------------------------------------------------------------------------
_ONES = ["సున్న", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"]
_TEENS = {
    10: "పది", 11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు", 14: "పద్నాలుగు",
    15: "పదిహేను", 16: "పదహారు", 17: "పదిహేడు", 18: "పద్దెనిమిది", 19: "పంతొమ్మిది",
}
_TENS = {
    20: "ఇరవై", 30: "ముప్పై", 40: "నలభై", 50: "యాభై",
    60: "అరవై", 70: "డెబ్బై", 80: "ఎనభై", 90: "తొంభై",
}
# forms used when more digits follow (e.g. "ఇరవై ఆరు")
_DIGIT_MAP = {str(i): chr(0x0C66 + i) for i in range(10)}


def _two_digits(n):
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n]
    tens, ones = (n // 10) * 10, n % 10
    return _TENS[tens] if ones == 0 else "%s %s" % (_TENS[tens], _ONES[ones])


def number_to_telugu(n):
    """Cardinal number -> Telugu words (0 <= n < 10**9)."""
    if n < 100:
        return _two_digits(n)
    parts = []
    crore, n = divmod(n, 10_000_000)
    lakh, n = divmod(n, 100_000)
    thousand, n = divmod(n, 1000)
    hundred, rest = divmod(n, 100)
    if crore:
        parts.append("కోటి" if crore == 1 else "%s కోట్ల" % number_to_telugu(crore))
    if lakh:
        parts.append("లక్ష" if lakh == 1 else "%s లక్షల" % number_to_telugu(lakh))
    if thousand:
        parts.append(("%s వేల" % _two_digits(thousand)) if thousand > 1 else "వెయ్యి")
    if hundred:
        parts.append("వంద" if hundred == 1 else "%s వందల" % _ONES[hundred])
    if rest:
        parts.append(_two_digits(rest))
    words = " ".join(parts)
    # trailing round numbers take the plural/noun form (e.g. రెండు వేలు, మూడు వందలు)
    if rest == 0:
        if hundred:
            words = re.sub(r"వందల$", "వందలు", words) if hundred > 1 else words
        elif thousand:
            words = re.sub(r"వేల$", "వేలు", words)
    return words


def _ordinal(word):
    """Very small ordinal builder: పన్నెండు -> పన్నెండవ."""
    if word.endswith("ై"):
        return word[:-1] + "య్యవ"
    if word.endswith("ు") or word.endswith("ి"):
        return word[:-1] + "వ"
    return word + "వ"


_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_ORD_RE = re.compile(r"(\d+)\s*వ(?=\s|$|[^ఀ-౿])")
_NUM_RE = re.compile(r"\d+")


def _to_ascii_digits(text):
    # Telugu digits U+0C66..U+0C6F -> ASCII
    return re.sub(r"[౦-౯]", lambda m: str(ord(m.group(0)) - 0x0C66), text)


def normalize_telugu(text):
    """NFC-normalise, expand times / ordinals / numbers, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch not in _IGNORED)
    text = _to_ascii_digits(text)
    # 10:45 -> "పది నలభై ఐదు"   (hour then minutes; 10:00 -> "పది")
    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        return number_to_telugu(h) if mi == 0 else "%s %s" % (number_to_telugu(h), number_to_telugu(mi))

    text = _TIME_RE.sub(_time, text)
    text = _ORD_RE.sub(lambda m: _ordinal(number_to_telugu(int(m.group(1)))), text)
    text = _NUM_RE.sub(lambda m: number_to_telugu(int(m.group(0))), text)
    text = re.sub(r"[“”\"‘’()\[\]{}<>*_/\\|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------------------
# Rule-based G2P
# ----------------------------------------------------------------------------
def word_to_phones(word):
    """Telugu word (single token, no punctuation) -> list of phones."""
    word = unicodedata.normalize("NFC", word)
    word = "".join(ch for ch in word if ch not in _IGNORED)
    phones = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in _CONSONANTS:
            phones.append(_CONSONANTS[ch])
            nxt = word[i + 1] if i + 1 < n else ""
            if nxt == _VIRAMA:
                i += 2
            elif nxt in _VOWEL_SIGNS:
                phones.extend(_VOWEL_SIGNS[nxt])
                i += 2
            else:
                phones.append("a")  # inherent vowel
                i += 1
        elif ch in _INDEPENDENT_VOWELS:
            phones.extend(_INDEPENDENT_VOWELS[ch])
            i += 1
        elif ch in _ANUSVARA:
            # homorganic nasal before a stop, plain /m/ otherwise or word-finally
            j = i + 1
            nxt_ph = _CONSONANTS.get(word[j]) if j < n else None
            phones.append(_NASAL_BEFORE.get(nxt_ph, "m"))
            i += 1
        elif ch == _VISARGA:
            phones.append("h")
            i += 1
        else:
            i += 1  # unknown symbol: skipped
    return phones


_PUNCT_SPLIT = re.compile(r"([,;.\-\?\!:\s])")


def telugu_to_phone_string(text):
    """Normalised or raw Telugu text -> '{k a ... sp ...}' string used by the model."""
    text = normalize_telugu(text)
    phones = []
    for tok in _PUNCT_SPLIT.split(text):
        if tok is None or tok == "" or tok.isspace():
            continue
        if re.fullmatch(r"[,;.\-\?\!:]", tok):
            phones.append("sp")
        else:
            phones.extend(word_to_phones(tok))
    return "{" + " ".join(phones) + "}"


def telugu_words(text):
    """Words of the normalised text with punctuation removed (for MFA .lab / dictionary)."""
    text = normalize_telugu(text)
    text = re.sub(r"[,;.\-\?\!:]", " ", text)
    return [w for w in text.split() if w]
