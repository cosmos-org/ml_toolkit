from abc import ABC, abstractmethod
from typing import List

class Opinionclassifier(ABC):
    classes = ['poll','review','ask','other']
    def __init__(self):
        pass
    @abstractmethod
    def opinion_classify(opinion:dict)->List:
        pass

class Keyword_Opinionclassifier(Opinionclassifier):

    def opinion_classify(opinion:dict)->List:
        pass

class ML_Opinionclassifier(Opinionclassifier):
    
    def opinion_classify(opinion:dict)->List:
        pass