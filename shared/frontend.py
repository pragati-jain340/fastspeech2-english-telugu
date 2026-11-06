"""The MMS training frontend: English words or uroman Telugu, no spaces."""
import re


def tokens_for(text, language):
    if any(c.isdigit() for c in text):
        raise ValueError('Digits require an explicit spoken form; pass --spoken-text. No digits are silently removed.')
    if language.lower() == 'telugu':
        from uroman import Uroman
        text = Uroman().romanize_string(text, lcode='tel')
    elif language.lower() != 'english':
        raise ValueError(language)
    elif any(c.isalpha() and not c.isascii() for c in text):
        raise ValueError('English frontend supports ASCII letters; provide an explicit spoken form.')
    tokens = list(''.join(re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())))
    if not tokens:
        raise ValueError('No alignable characters')
    return tokens


def encode(text, language, metadata):
    tokens = tokens_for(text, language)
    missing = sorted(set(tokens) - set(metadata['vocabulary']))
    if missing:
        raise ValueError(f'Characters absent from training vocabulary: {missing}')
    return tokens, [metadata['vocabulary'][c] for c in tokens]
