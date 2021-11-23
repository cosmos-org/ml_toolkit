from .relation_extractor import Keyword_Topic_Opinion_Relation_Extractor

class ToolKit_Loader():
    relation_extractor_map = {
        "1" : Keyword_Topic_Opinion_Relation_Extractor
    }
    def __init__(self):
        pass
    @staticmethod
    def load_relation_extractor(type = 1):
        return ToolKit_Loader.relation_extractor_map[type].default_init()
