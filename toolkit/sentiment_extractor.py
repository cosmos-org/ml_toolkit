from abc import ABC, abstractmethod
from typing import List
from nltk import sent_tokenize
from util import *
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification
import pandas as pd
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
from vncorenlp import VnCoreNLP
from .entity_class import Topic, Opinion
from .data import Doc, Sentiment, TextBlob
annotator = VnCoreNLP("vncore/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')

softmax = nn.Softmax(dim=1)
model_path = 'model_sentiment_bert'

model_sentiment = AutoModelForSequenceClassification.from_pretrained(model_path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
                                       
tokenizerbert = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

class Sentimentor(ABC):
    def __init__(self) -> None:
        super().__init__()
    def extract_sentiment(opinion:dict,topic:List)->List:
        pass
    
class Bert_Sentimentor(Sentimentor):
    MAX_LENGTH = 256
    def __init__(self) -> None:
        super().__init__()
        self.model_name = 'BERT_bank_v1.4'
        self.model = model_sentiment
        self.device = device
        self.tokenizer = tokenizerbert
        self.batch_size = 4
    @classmethod
    def default_init(cls):
        return cls() 
    def sentiment_extract(self,opinion:dict = {},topic:List = [])->List:
        def get_final_label(neg:float=0.0, pos:float=0.0, neu:float=0.0, score=0):
            if neg > pos and neg > neu:
                return "NEG"
            elif pos > neg and pos > neu:
                return "POS"
            return "NEU"
        def get_final_score(neg:float=0.0, pos:float=0.0, neu:float=0.0, score=0):
            if neg > pos and neg >= neu:
                return neg
            elif pos > neg and pos >= neu:
                return pos
            else:
                return neu
        topic = Topic(topic)
        opinion = Opinion.from_dict(opinion)
        content = opinion.content
        _, doc  = self.predict_doc_sentiment(content)
        # _, doc =  predict_bert("bank_v4",domain_id, title, snippet, content)
        final_label = get_final_label(**{**doc.global_sentiment})
        
        final_score = get_final_score(**{**doc.global_sentiment})  
        return {'label':final_label, 'score': final_score}
        
        
    def eval_subject_sentiment(self,sentiment_sentences, length_doc, num_threshold_negative=2):
        # sentence: sentence, subject, score, label
        dict_subject_sentiment = {}
        for sentence in sentiment_sentences:
            subject = sentence["subject"]
            if sentence["subject"] not in dict_subject_sentiment.keys():
                dict_subject_sentiment[subject] = {}
                dict_subject_sentiment[subject]["sentiment"] = "neutral"
                dict_subject_sentiment[subject]["num_neg"] = []
                dict_subject_sentiment[subject]["num_pos"] = []
                dict_subject_sentiment[subject]["num_neu"] = []
                dict_subject_sentiment[subject]["lst_sentence"] = []
            dict_subject_sentiment[subject]["lst_sentence"].append(sentence)
            if sentence["label"] == -1:
                dict_subject_sentiment[subject]["num_neg"].append(sentence)
            if sentence["label"] == 1:
                dict_subject_sentiment[subject]["num_pos"].append(sentence)
            if sentence["label"] == 0:
                dict_subject_sentiment[subject]["num_neu"].append(sentence)
        for subject, dict in dict_subject_sentiment.items():

            neg_sentence_ratio = len(dict["num_neg"]) / length_doc
            pos_sentence_ratio = len(dict["num_pos"]) / length_doc
            
            # if neg_sentence_ratio >= 0.3 or (neg_sentence_ratio >=0.15 and pos_sentence_ratio < 2* neg_sentence_ratio) or (len(dict["num_neg"]) == 1 and len(sentiment_sentences) < 6)\
            #         or (len(dict["num_neg"]) == 1 and len(dict["num_pos"]) <= 2):
            if len(dict["num_neg"]) >= num_threshold_negative or (len(dict["num_neg"]) == 1 and len(sentiment_sentences) < 6)\
                    or (len(dict["num_neg"]) == 1 and len(dict["num_pos"]) <= 2):
            # if (neg_sentence_ratio  >= 0.2 and length_doc <= 8) or ( neg_sentence_ratio  >= 0.25 and 
            #         length_doc > 8 and neg_sentence_ratio > 0.5 * pos_sentence_ratio) or neg_sentence_ratio > 0.3:
                dict["sentiment"] = "negative"
                dict["score_sentiment"] = {}
                dict["score_sentiment"]["neg"] = sum([s["score"]["neg"] for s in dict["num_neg"]])/len(dict["num_neg"])
                dict["score_sentiment"]["pos"] = sum([s["score"]["pos"] for s in dict["num_neg"]])/len(dict["num_neg"])
                dict["score_sentiment"]["neu"] = sum([s["score"]["neu"] for s in dict["num_neg"]])/len(dict["num_neg"])
            elif (len(dict["num_neg"]) == 1 and len(dict["num_pos"]) > 2)\
                    or (len(dict["num_neg"]) == 0 and len(dict["num_pos"]) > 2)\
                            or (len(dict["num_neg"]) == 0 and len(dict["num_pos"]) <=2 and len(dict["num_pos"]) and len(sentiment_sentences) < 5):
            # elif pos_sentence_ratio > 0.3 or (len(dict["num_neg"]) == 1 and len(dict["num_pos"]) > 2)\
            #         or (len(dict["num_neg"]) == 0 and len(dict["num_pos"]) > 2)\
            #                 or (len(dict["num_neg"]) == 0 and len(dict["num_pos"]) <=2 and len(dict["num_pos"]) and len(sentiment_sentences) < 5):
            # elif (pos_sentence_ratio > 0.3 and neg_sentence_ratio < 0.5 * pos_sentence_ratio):    
                dict["sentiment"] = "positive"
                dict["score_sentiment"] = {}
                dict["score_sentiment"]["neg"] = sum([s["score"]["neg"] for s in dict["num_pos"]])/len(dict["num_pos"])
                dict["score_sentiment"]["pos"] = sum([s["score"]["pos"] for s in dict["num_pos"]])/len(dict["num_pos"])
                dict["score_sentiment"]["neu"] = sum([s["score"]["neu"] for s in dict["num_pos"]])/len(dict["num_pos"])
            else:
                dict["sentiment"] = "neutral"
                dict["score_sentiment"] = {}
                dict["score_sentiment"]["neg"] = sum([s["score"]["neg"] for s in dict["num_neu"]])/max(1, len(dict["num_neu"]))
                dict["score_sentiment"]["pos"] = sum([s["score"]["pos"] for s in dict["num_neu"]])/max(1, len(dict["num_neu"]))
                dict["score_sentiment"]["neu"] = sum([s["score"]["neu"] for s in dict["num_neu"]])/max(1, len(dict["num_neu"]))
        return dict_subject_sentiment

    def predict_doc_sentiment(self,content:str=''): 
        doc = Doc(domain_id = '', title = '', snippet = '', content=content)
        length_doc = len(sent_tokenize(content))
        sentences = get_sentence('', '', content, '')
        # if not sentences:
        #     return normalize_output(doc), doc   
        lst_sentences = [s[0] for s in sentences]
        lst_subjects = [s[1] for s in sentences]
        result = self.predict_with_bert( lst_sentences, lst_subjects, self.device)
        result = self.eval_subject_sentiment(result, length_doc)
        # print(result)
        lst_sents_of_Doc = []   
        domain_sentiment = {}
        for subject, value in result.items():
            domain_sentiment[subject] = Sentiment(neg=value["score_sentiment"]["neg"], neu=value["score_sentiment"]["neu"],\
                                                    pos=value["score_sentiment"]["pos"])
            for sentence in value["lst_sentence"]:
                score = sentence["score"]
                t = TextBlob(sentence["sentence"])
                t.subjects = [sentence["subject"]]
                senti = Sentiment(neg=score["neg"], neu=score["neu"], pos=score["pos"])
                t.model_sentiment = senti
                t.text = sentence["sentence"]
                t.tokens = sentence["sentence"].split()
                lst_sents_of_Doc.append(t)
        doc.sentences = lst_sents_of_Doc
        doc.domain_sentiment = domain_sentiment
        doc.global_sentiment = Sentiment(neu=0.0, neg=0.0, pos=0.0)
        neu_score = 0.0
        neg_score = 0.0
        pos_score = 0.0
        sum_len = 0
        for subject, res in result.items():
            if res["sentiment"] == "negative":
                neu_score += len(res["num_neg"]) * res["score_sentiment"]["neu"]
                neg_score += len(res["num_neg"]) * res["score_sentiment"]["neg"]
                pos_score += len(res["num_neg"]) * res["score_sentiment"]["pos"]
                score = 10 * len(res["num_neg"]) * res["score_sentiment"]["neg"] / length_doc
                doc.domain_sentiment[subject].score = score
                sum_len += len(res["num_neg"])
            elif res["sentiment"] == "positive":
                neu_score += len(res["num_pos"]) * res["score_sentiment"]["neu"]
                neg_score += len(res["num_pos"]) * res["score_sentiment"]["neg"]
                pos_score += len(res["num_pos"]) * res["score_sentiment"]["pos"]
                score = 10 * len(res["num_pos"]) * res["score_sentiment"]["pos"] / length_doc
                doc.domain_sentiment[subject].score = score
                sum_len += len(res["num_pos"])
            else:
                neu_score += len(res["num_neu"]) * res["score_sentiment"]["neu"]
                neg_score += len(res["num_neu"]) * res["score_sentiment"]["neg"]
                pos_score += len(res["num_neu"]) * res["score_sentiment"]["pos"]
                score = 10 * len(res["num_neu"]) * res["score_sentiment"]["neu"] / length_doc
                doc.domain_sentiment[subject].score = score
                sum_len += len(res["num_neu"])
        doc.global_sentiment.neu = neu_score/max(sum_len, 1)
        doc.global_sentiment.neg = neg_score/max(sum_len, 1)
        doc.global_sentiment.pos = pos_score/max(sum_len, 1)
        # print('doccc')
        # print(doc)
        return self.normalize_output(doc), doc
    def normalize_output(self,doc):
        result = {
            'benchmark_id': 0,
            'mse_ver': '6.0.0',
            'idxcol': 0,
            'children_id': 0,
            'msg': '',
            'domain_id': doc["domain_id"]
        }

        domain_id = doc["domain_id"]
        title = doc["title"]
        snippet = doc["snippet"]
        content = doc["content"]
        sentiment = {"global_label": "NEU", 
                    "domain_senti": {"score": 0.0, "label": "NEU"}, 
                    "final_subject_result": []}
        for domain, senti in doc["domain_sentiment"].items():
            d = {"subject_name": domain, "subject_id": '', "label": "NEU"}
            neg, neu, pos = senti.neg, senti.neu, senti.pos
            if neg > neu and neg > pos:
                d["label"] = "NEG"
            if pos > neu and pos > neg:
                d["label"] = "POS"
            sentiment["final_subject_result"].append(d)
        global_senti = doc["global_sentiment"]
        neg_g, neu_g, pos_g = global_senti.neg, global_senti.neu, global_senti.pos
        if neg_g > neu_g and neg_g > pos_g:
            sentiment["global_label"] = "NEG"
            sentiment["domain_senti"]["score"] = neg_g
            sentiment["domain_senti"]["label"] = "NEG"
        if pos_g > neu_g and pos_g > neg_g:
            sentiment["global_label"] = "POS"
            sentiment["domain_senti"]["score"] = pos_g
            sentiment["domain_senti"]["label"] = "POS"
        if neu_g > pos_g and neu_g > neg_g:
            sentiment["domain_senti"]["score"] = neu_g
        
        result["sentiment"] = sentiment
        return result

    def predict_with_bert(self, sentences_pre, lst_subjects, device):
        sentences = [remove_punctuation(text) for text in sentences_pre]
        lst_input = []
        for text in sentences:
            if len(text) > 0:
                try:
                    text = annotator.tokenize(text)[0]
                    # text = T.word_tokenize(text, tokenize_option=0)
                except IndexError:
                    print(text)
                if self.model_name == "BERT":
                    lst_input.append(" ".join(text).lower())
                else:
                    lst_input.append(" ".join(text))
            else:
                lst_input.append("NULL")
        # print(lst_input)
        test_data = self.tokenizer(lst_input, padding="max_length", truncation=True, max_length=Bert_Sentimentor.MAX_LENGTH)
        df = pd.DataFrame({"input_ids": test_data.input_ids, "attention_mask": test_data.attention_mask})
        test_dataset = Dataset.from_pandas(df)
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"]
        )
        test_loader = DataLoader(TensorDataset(test_dataset["input_ids"], test_dataset["attention_mask"]), \
                                batch_size=self.batch_size, shuffle=False)
        prediction, probability = self.predict( test_loader, device)
        lst_result = []
        for i in range(len(sentences)):
            lst_result.append({"sentence": sentences_pre[i], "label": int(prediction[i]), "score": {"neg": float(probability[i][0]),\
                "neu": float(probability[i][1]), "pos": float(probability[i][2])}, "subject": lst_subjects[i]})
        return lst_result
    def predict(self,test_loader, device):
        self.model.eval()
        self.model.to(device)
        prediction = []
        probability = []
        for batch in test_loader:
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
            probs = outputs[0]
            probs = softmax(probs)
            probs = probs.detach().cpu().numpy()
            pred_labels = np.argmax(probs, axis=1)
            prediction.extend(pred_labels)
            if self.model_name == "BERT":
                probs = [[prob[2], prob[0], prob[1]] for prob in probs]
            else:
                probs = [[prob[0], prob[1], prob[2]] for prob in probs]
            probability.extend(probs)
        prediction = np.array(prediction)
        if self.model_name == "BERT":
            prediction[prediction==2] = -1
        else:
            prediction = [ x-1 for x in prediction]     
        return list(prediction), list(probability)

                                


                       