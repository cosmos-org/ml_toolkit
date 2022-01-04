
from typing import *

from dataclasses import dataclass, asdict, field




class Topic():
    def __init__(self, keywords_list: List = [],):
        self.keywords_list = keywords_list
        
class Opinion():
    def __init__(self, type : str = 'review',content: str = '', content_type:str = '',dictionary : dict = {}):
        self.type = type
        self.content = content
        self.content_type = content_type
        self.dictionary = dictionary

    @classmethod
    def from_dict(cls, dictionary: dict = {}):
        type = dictionary.get('type','')
        content = dictionary.get('content','')
        content_type = dictionary.get('content_type','review')
        return cls(type,content,content_type,dictionary)
