import re

at_words = re.compile("@[0-9]+")

def features_from_words(sentence, i):
    
    word = sentence[i][0]
    postag = sentence[i][1]
    
    word_features = {
        'bias': 5.0,
        'lower_word': word.lower(),
        'isupper': word.isupper(),
        'suf_3': word[-3:],
        'suf_2': word[-2:],
        'containshash': "#" in word ,
        'issq': ("sq" == word),
        'isft': ("ft" == word),
        'is@': at_words.fullmatch(word) != None,
        'islink': "http" in word,
        'istitle': word.istitle(),
        'isdigit': word.isdigit(),
        'postag': postag
    }
    if(i>0):
        word1 = sentence[i-1][0]
        postag1 = sentence[i-1][1]
        
        word_features.update({
            '-1lower': word1.lower(),
            '-1istitle': word1.istitle(),
            '-1isupper': word1.isupper(),
            '-1postag': postag1
        })
    else:
        word_features['begin'] = True

    if(i<len(sentence)-1):
        
        word1 = sentence[i+1][0]
        postag1 = sentence[i+1][1]
        
        word_features.update({
            '+1lower': word1.lower(),
            '+1istitle': word1.istitle(),
            '+1isupper': word1.isupper(),
            '+1postag': postag1
        })
    else:
        word_features['end'] = True

    return word_features


def features_from_sentence(sentence):
    return [features_from_words(sentence, i) for i in range(len(sentence))]

def process_token(tokens):
    return [features_from_sentence(i) for i in tokens]