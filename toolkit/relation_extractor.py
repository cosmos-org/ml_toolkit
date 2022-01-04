from abc import ABC, abstractmethod
from typing import List
import re
import numpy as np
import traceback
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from vncorenlp import VnCoreNLP
from .entity_class import Topic, Opinion
class Topic_Opinion_Relation_Extractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def relation_extract(self,topic:List, opinion:dict = {})->float:
        pass

class Keyword_Topic_Opinion_Relation_Extractor(Topic_Opinion_Relation_Extractor):
    def __init__(self,segmentor,word_vectorizer,doc_vectorizer):
        super(Keyword_Topic_Opinion_Relation_Extractor, self).__init__()
        self.segmentor = segmentor
        self.word_vectorizer = word_vectorizer
        self.doc_vectorizer = doc_vectorizer
    @classmethod
    def default_init(cls):
        doc_vectorizer_path = 'doc_vectorizer/vn_tfidf_builder.pkl'
        word_vectorizer_path = 'word_vectorizer/w100.pkl'
        segmentor_path = 'vncore/VnCoreNLP-1.1.1.jar'
        segmentor = VnCoreNLP(segmentor_path, annotators="wseg", max_heap_size='-Xmx2g')
        with open(word_vectorizer_path,'rb') as f:
            word_vectorizer = pickle.load(f)
        with open(doc_vectorizer_path,'rb') as f:
            doc_vectorizer = pickle.load(f)
        return cls(segmentor,word_vectorizer,doc_vectorizer)
    def extrac_score(self,keywords,content):
        score = 0
        for k in keywords:
            if k.lower() in content.lower():
                score += 0.1
        return score
    def relation_extract(self,topic:List, opinion:dict = {})->float:
        topic = Topic(topic)
        opinion = Opinion.from_dict(opinion)
        content = opinion.content
        keywords = topic.keywords_list
        numkey = 5
        return {'score': self.calculate_score(content,keywords,numkey)+ self.extrac_score(keywords,content)}
    def chunk2chunk_similarity(self,ls1:list,ls2:list,weight1:list,weight2:list):

        #['','',....] -> [[], [], ...] : 
        def get_word_vectors(words:list,weights):
        
            vectors = []
            w = []
            for ind,word in enumerate(words):
                try:
                    vectors.append(self.word_vectorizer[word])
                    w.append(float(weights[ind]))
                except:
                    traceback.print_exc()
                    print('Loi w2v voi tu: - {}'.format(word))
                    pass
            
            return vectors, w
        # print(ls1)
        # print(ls2)
        
        
        v1s, weight1 = get_word_vectors(ls1,weight1)
        v1s = np.array(v1s)
        weight1 = np.array(weight1)
        v2s, weight2 = get_word_vectors(ls2,weight2)

        v2s = np.array(v2s)
        weight2 = np.array(weight2)

        cos_matrix = cosine_similarity(v1s,v2s)

        s = np.ones(cos_matrix.shape)
        if (weight1 != None).all():
            for ind,w in enumerate(weight1):
                cos_matrix[ind] = np.multiply(np.array(w),cos_matrix[ind])
                s[ind] = np.multiply(np.array(w),s[ind])
        
        if (weight2 != None).all():
            for ind,w in enumerate(weight2):
                cos_matrix[:,ind] = np.multiply(np.array(w),cos_matrix[:,ind])
                s[:,ind] = np.multiply(np.array(w),s[:,ind])
        sim = np.sum(cos_matrix)
        divide = np.sum(s)
        if (float(divide) == 0):
            return 0
        return float(sim/divide)

    
    def extract_keyword_from_document_and_weight(self,doc:str,top:int):
        top = int(top)
        if top > len(doc.split(' ')):
            print('Numkey > len(doc)')
            top  = len(doc.split(' '))
       
        def inds_to_words(vocab,inds:list):
            res = []
            for i in inds:    
                res.append(list(vocab.keys())[list(vocab.values()).index(i)])
            return res
    
        cvector = self.doc_vectorizer.transform([doc]).toarray()
    

        top_index = cvector[0].argsort()[-top:][::-1]
        top_tfidfs = np.sort(cvector[0])[-top:][::-1]
        top_tfidfs = top_tfidfs.tolist()
        
        top_words = inds_to_words(self.doc_vectorizer.vocabulary_,top_index)

        l1 = []
        l2 = []
        for i in range(len(top_tfidfs)):
            if top_tfidfs[i] > 0.0:
                l1.append(top_words[i])
                l2.append(top_tfidfs[i])

        return l1,l2
    def keywords2doc_sim(self,keywords:list,doc:str,num_keyword2):

        top_words_doc,topscore2 = self.extract_keyword_from_document_and_weight(doc,num_keyword2)

        topscore1 = np.ones([len(keywords)]).tolist()
        print(keywords,top_words_doc,topscore1,topscore2)
        return self.chunk2chunk_similarity(keywords,top_words_doc,topscore1,topscore2)

    def segment(self,text:str)->str:
        message = text
        message = message.replace('_',' ')
        segmented_message = ''
        segmented_sentences = self.segmentor.tokenize(message)
        for sentence in segmented_sentences:
            segmented_message += ' ' + ' '.join(sentence)
        segmented_result = segmented_message
        segmented_result = re.sub('\s+',' ',segmented_result).strip()
        return segmented_result

    def calculate_score(self,text,keywords,numkey2)->float:
        text = self.segment(text)
        print(text)
        segmented_keywords = []
        for k in keywords:
            segmented_keywords.extend(self.segment(k).split(' '))
        # print(keywords)
        overall_sim = self.keywords2doc_sim(segmented_keywords,text,numkey2)
        return overall_sim
    
