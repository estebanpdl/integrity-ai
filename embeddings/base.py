# -*- coding: utf-8 -*-

'''
Defines VectorModel base class

'''

# import modules
import os

# import abstract base class
from abc import ABC, abstractmethod

# VectorModel abstract base class
class VectorModel(ABC):
    '''
    VectorModel abstract base class
    '''
    def __init__(self):
        '''
        Initialize VectorModel abstract base class
        '''
        self.log_file = './logs/vector_model_log.txt'
    
    def _log_write(self, message: str) -> None:
        '''
        Write a message to the log file.
        '''
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    @abstractmethod
    def compute_embeddings(self, data: list[str]) -> list[list[float]]:
        '''
        Compute embeddings for a list of strings.
        '''
        pass

