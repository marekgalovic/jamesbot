import re
import nltk

def normalize_token(token, replace_digits=True):
    m = re.match('(\d+(?:\.?\d+)?)', token)
    if m is not None:
        prefix = token[:m.start()]
        if len(prefix) > 0:
            yield prefix
        
        if replace_digits:
            yield '.DIGIT'
        else:
            yield m.group(1)
            
        suffix = token[(m.start()+len(m.group(1))):]
        if len(suffix) > 0:
            yield suffix
    elif token[:5] == '.slot':
        yield str(token[:5].upper()) + token[5:]
    else:
        yield token

def tokenize(text, replace_digits=True):
    tokens = nltk.word_tokenize(str(text).lower())
    return [normalized_token for token in tokens for normalized_token in normalize_token(token, replace_digits)]
