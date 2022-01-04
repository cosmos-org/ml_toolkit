from .relation_extractor import Keyword_Topic_Opinion_Relation_Extractor
from .sentiment_extractor import Bert_Sentimentor
class ToolKit_Loader():
    relation_extractor_map = {
        "1" : Keyword_Topic_Opinion_Relation_Extractor
    }
    sentiment_map = {
        "1" : Bert_Sentimentor 
    }
    def __init__(self):
        pass
    @staticmethod
    def load_relation_extractor(type = 1):
        return ToolKit_Loader.relation_extractor_map[str(type)].default_init()
    @staticmethod
    def load_sentiment_extractor(type = 1):
        return ToolKit_Loader.sentiment_map[str(type)].default_init()
