# -*- coding: utf-8 -*-

'''
LLM as a Judge: Scoring Rubric
'''

# import modules
import tomli
import string

# import LLM base class
from models import LanguageModel

# LLM as a Judge class
class LLMJudge:
    '''
    LLMJudge class
    '''
    def __init__(self, llm_engine: LanguageModel, args: dict = None):
        '''
        '''
        pass