# -*- coding: utf-8 -*-

'''
Defines the abstract base class

'''

# import modules
from abc import ABC, abstractmethod

# LanguageModel abstract base class
class LanguageModel(ABC):
    '''
    LanguageModel abstract base class
    '''
    def __init__(self):
        '''
        Initialize LanguageModel abstract base class
        '''
        pass

    @abstractmethod
    def _estimate_tokens(self, prompt: str) -> int:
        '''
        Estimate the number of tokens in the prompt.

        :param prompt: The prompt to be estimated.
        :return: The estimated number of tokens.
        '''
        pass

    @abstractmethod
    def run_parallel_prompt_tasks(self, messages: list) -> None:
        '''
        Process multiple messages in a single call.

        :param messages: List of messages to be processed.
        '''
        pass
    