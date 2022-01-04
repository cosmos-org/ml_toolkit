from typing import *

from dataclasses import dataclass, asdict, field


class Subcriptable:

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        for key in self.__annotations__:
            yield key

    def keys(self):
        return self.__iter__()

    def values(self):
        for key in self.__annotations__:
            yield getattr(self, key)

    def items(self):
        for key in self.__annotations__:
            yield key, getattr(self, key)

    def update(self, data:Dict[str, Any]={}, **kwargs):
        data.update(kwargs)
        for key, value in data.items():
            self.__setitem__(key, value)


@dataclass
class Sentiment(Subcriptable):
    neg:float = 0.0
    neu:float = 0.0
    pos:float = 0.0
    score:float = 0.0


@dataclass
class TextBlob(Subcriptable):
    text:str
    tokens:List[str] = field(default_factory=list)
    subjects:Dict[str, float] = field(default_factory=dict)

    model_sentiment:Sentiment = field(default_factory=Sentiment)


@dataclass
class Doc(Subcriptable):
    domain_id:str = ''
    title:str = ''
    snippet:str = ''
    content:str = ''
    sentences:List[TextBlob] = field(default_factory=list)

    subjects:Dict[str, Dict[str, str]] = field(default_factory=dict)

    model_sentiment:Dict[str, Sentiment] = field(default_factory=dict)
    domain_sentiment:Dict[str, Sentiment] = field(default_factory=dict)

    global_sentiment:Sentiment = field(default_factory=Sentiment)
