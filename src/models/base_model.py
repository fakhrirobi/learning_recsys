from abc import ABC, abstractmethod
import scipy.sparse as sp 

class RecommenderBase(ABC) : 
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def fit(self) : 
        pass 
    
    @abstractmethod
    def predict(self) : 
        pass 
    
    @abstractmethod
    def recommend(self): 
        pass 
    
    

        