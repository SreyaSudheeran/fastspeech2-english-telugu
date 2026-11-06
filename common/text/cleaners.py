""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''


# Regular expression matching whitespace:
import re
from unidecode import unidecode
from .numbers import normalize_numbers
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


_time_re = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})\s*([AaPp])\.?[Mm]\.?(?![A-Za-z])|(?<!\d)(\d{1,2}):(\d{2})(?!\d)")


def _expand_time(m):
    """10:45 AM -> 'ten forty-five a m'; 9:05 -> 'nine oh five'; 7:00 -> "seven o'clock"."""
    import inflect

    eng = inflect.engine()
    if m.group(1) is not None:
        h, mi, mer = int(m.group(1)), int(m.group(2)), m.group(3).lower() + " m"
    else:
        h, mi, mer = int(m.group(4)), int(m.group(5)), ""
    out = eng.number_to_words(h, andword="")
    if mi == 0:
        out += " o'clock" if not mer else ""
    elif mi < 10:
        out += " oh " + eng.number_to_words(mi, andword="")
    else:
        out += " " + eng.number_to_words(mi, andword="")
    return (out + " " + mer).strip()


def expand_time(text):
    return re.sub(_time_re, _expand_time, text)


def english_cleaners(text):
    '''Pipeline for English text: ASCII, clock times, lowercase, numbers, abbreviations.
    (Modified: clock-time / AM-PM expansion; hyphens in expanded numbers become spaces
    so that "forty-five" is not read with a pause.)'''
    text = convert_to_ascii(text)
    text = expand_time(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = text.replace("-", " ")
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def telugu_cleaners(text):
    '''Pipeline for Telugu text: NFC, digit/time/ordinal expansion to Telugu words.'''
    from .telugu import normalize_telugu

    return normalize_telugu(text)
