import math, re

_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def words(text):
    return [w.lower() for w in _word_re.findall(text)]

def ttr(text):
    """Type-token ratio: number of unique words / total number of words."""
    ws = words(text)
    return len(set(ws)) / max(len(ws), 1)
