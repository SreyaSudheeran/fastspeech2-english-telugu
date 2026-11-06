""" from https://github.com/keithito/tacotron (modified)

Symbol table, selected per language through the FS2_LANG environment variable
("en" or "te"). Every language entry script (English/*.py, Telugu/*.py) sets
FS2_LANG before importing anything else, so each model gets its own vocabulary.

Changes w.r.t. upstream FastSpeech2:
  * the Mandarin/pinyin symbols are removed;
  * a Telugu phone inventory (text/telugu.py) is added;
  * the table is language-specific instead of a union of all languages.
"""
import os

from text import cmudict

LANG = os.environ.get("FS2_LANG", "en").lower()

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

if LANG == "te":
    from text.telugu import TELUGU_PHONES

    _phones = ["@" + s for s in TELUGU_PHONES]
elif LANG == "en":
    # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters)
    _phones = ["@" + s for s in cmudict.valid_symbols]
else:
    raise ValueError("FS2_LANG must be 'en' or 'te', got %r" % LANG)

symbols = (
    [_pad] + list(_special) + list(_punctuation) + list(_letters) + _phones + _silences
)
