

# workflow: đầu vào là văn bản, và domain
# từ domain: lấy ra alias từ khóa với domain
# for mỗi câu trong bài tìm ra các câu chứa subject

import re
import unicodedata
import bs4
import nltk
# nltk.download('punkt')
from nltk import sent_tokenize

import string




def del_html(doc):
    soup = bs4.BeautifulSoup(doc, features="html.parser")
    return soup.get_text(' ')

def del_link(text):
    link = r'http[\S]*'
    text = re.sub(link, ' ', str(text))
    return text

def normalize_text(text):
    # text = text.lower()
    text = text.replace(u'"', u' ')
    text = text.replace('🏻','')
    return text

def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = del_html(text)
    text = del_link(text)
    text = normalize_text(text)
    return text

# def remove_punctuation(text):
#     translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
#     text = text.translate(translator)
#     return text

def filter_sent(text: str, keywords: dict, kp: None, max_leng_of_paragraph:int=180):
    text = clean_text(text)
    lst_sent = []
    sentences = sent_tokenize(text)
    for i, sent in enumerate(sentences):
        labels = kp.extract_keywords(sent)
        labels = list(set(labels))
        if len(labels) != 0:
            if i == 0 and len(sentences) > 1:
                ss = ' '.join([sent, sentences[i+1]])
            elif i == 0 and len(sentences) == 1:
                ss = sent
            elif i > 0 and i < len(sentences)-1:
                ss = ' '.join([sentences[i-1], sent, sentences[i+1]]) 
            else:
                ss = ' '.join([sentences[i-1], sent])
            for label in labels:
                lst_sent.append((ss, label))
    result = []
    for sent in lst_sent:
        sentence = sent[0]
        subject = sent[1]
        if len(sentence.split()) < max_leng_of_paragraph:
            result.append(sent)
        else:
            for word in keywords[subject]:
                if sentence.find(word) != -1:
                    pos = sentence.find(word)
                    previous_of_sent = sentence[:pos]
                    after_of_sent = sentence[pos:]
                    len_previous = len(previous_of_sent.split())
                    len_after = len(after_of_sent.split())
                    if len_after < 15:
                        sentence_cut = sentence.split()[-max_leng_of_paragraph:]
                        sentence_cut = ' '.join(sentence_cut)
                    else:
                        if len_previous > 140 and len_after < 40:
                            min = len_previous - 140
                            sentence_cut = ' '.join(sentence.split()[min:])
                        elif len_previous > 140 and len_after > 40:
                            min = len_previous - 140
                            max = len_previous + 40
                            sentence_cut = ' '.join(sentence.split()[min:max])
                        else:
                            sentence_cut = sentence 
                    result.append((sentence_cut, subject))
                    break
    return result

def filter_sent_general(text):
    result = []
    text = clean_text(text)
    sentences = sent_tokenize(text)
    if len(sentences) > 2:
        for i in range(0, len(sentences)-2):
            result.append((' '.join(sentences[i:i+3]), "general"))
    else:
        result.append((' '.join(sentences), "general"))
    return result

def get_sentence(title, snippet, content, domain_id):
    sentences = []
    if domain_id == '':
        for text in [title, snippet, content]:
            if text.strip() != '':
                sentences.extend(filter_sent_general(text))
        return sentences
    
    # keywords = get_alias(domain_id)
    # keywords = []
    # kp = KeywordProcessor()
    # kp.add_keywords_from_dict(keywords)
    # for text in [title, snippet, content]:
    #     sentences.extend(filter_sent(text, keywords, kp))
    # return sentences # [(sent, subject)]
    
def remove_punctuation(text):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    return text
