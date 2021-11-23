from abc import ABC, abstractmethod
from typing import List

class Sentimentor(ABC):
    def __init__(self) -> None:
        super().__init__()
    def extract_sentiment(opinion:dict,topic:List)->List:
        pass