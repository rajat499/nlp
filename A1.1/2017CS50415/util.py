import pandas as pd
import numpy as np
import time
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

classes = {"Meeting and appointment": 1, 
           "For circulation": 2, 
           "Selection committe issues": 3, 
           "Policy clarification/setting": 4, 
           "Recruitment related": 5, 
           "Assessment related": 6,
           "Other": 7}

def convert_class(df):
    df.Class = df.Class.apply(lambda x: classes[x])
    return df

months = set(["jan", "feb", "mar", "apr", "may", "june", "july", "aug", "sep", "oct", "nov", "dec",
         "january", "february", "march", "april", "august", "september", "october", "november", "december"])

days = set(["mon", "tue", "wed", "thu", "fri", "sat", "sun",
       "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])

html_content = set(["br", "html", "gt", "body", "head",
                    'meta', 'content', 'text', 'html', 'charset', 'iso', 'style', 'type', 'text', 
                    'css', 'id', 'owaparastyle', 'style', 'head', 'body', 'fpstyle', 'ocsi', 'div', 
                    'style', 'direction', 'ltr', 'font', 'family', 'tahoma',
                    'span', 'style', 'font', 'family', 'verdana', 'color', 
                    '#000000', 'font', 'size', '10pt', 'div', "nbsp",
                   'class', 'plaintext'])

email_contents = ["fwd", "re", "fw", "dear", "regards", "sir", "madam"]
stop_words = set(stopwords.words('english')).union(months).union(days).union(html_content).union(email_contents)
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('stopwords')

def stop_words_removal(l):
    if(type(l) != list):
        return ""
    final = []
    for x in l:
        if(len(x)<2 or len(x)>42):
            continue
        if(x in stop_words):
            continue
        final.append(lemmatizer.lemmatize(x))
    return " ".join(final)

def preprocess(df, column):
    
    df[column] = df[column].str.lower()
    
    df[column] = df[column].str.replace("=\n", "")
    
    links = [r"http\S+", r"=.."]
    for l in links:
        df[column] = df[column].str.replace(l, " ", regex=True)
    
    space = ["<br>", ">", "<", "_", "(", ")", "&", ",", ":", "*", "'", 
             '"', "--", "-", "+", "\.\.", "|", ";", "[", "]", "â"]
    for s in space:
        df[column] = df[column].str.replace(s, " ")     
    
    pats = [r"\d+\.", r"\d+th", r"\d+st", r"(\s|/)\d+(\s|/)", r"(\s|/)\d+(\s|/)", r"(\s|/)\d+(\s|/)"
            , r"\sam\s", r"\spm\s", r'[^\x00-\x7f]']
    for p in pats:
        df[column] = df[column].str.replace(p, " ", regex=True)
    
    special= [" i ", " ii ", " iii ", " iv ", " v "]
    for s in special:
        df[column] = df[column].str.replace(s, " ") 
    
    df[column] = df[column].str.strip()
    df[column] = df[column].str.split(r'\s*[@.!:/%;,\s?\-]\s*')
    df[column] = df[column].apply(lambda x: stop_words_removal(x))
    return df

def oversample(df):
    
    counts = df.groupby(['Class']).count().Subject
    cls = df.Class.unique()
    
    new_train = pd.DataFrame(columns=df.columns)
    maximum = max(counts)

    for c in cls:
        data = df[df.Class == c]
        rep = maximum//counts.loc[c]
        new_train = pd.concat([new_train, pd.concat([data]*rep)])
        rep = maximum%counts.loc[c]
        if(rep==0):
            continue
        data = data.head(rep)
        new_train = pd.concat([new_train, data], axis=0)

    new_train.index = range(len(new_train))
    
    return new_train
