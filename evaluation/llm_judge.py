# -*- coding: utf-8 -*-

'''
LLM as a Judge: Scoring Rubric
'''

# import modules
import json
import time
import tomli
import string
import random
import threading

# LLM engines
from models import *

# LLM as a Judge class
class LLMJudge:
    '''
    LLMJudge class
    '''
    def __init__(self, args: dict = None):
        '''
        Initialize the LLMJudge class.
        '''
        # get arguments
        args = args or {}
        self.args = args

    def _load_system_prompt(self, language: str = 'en') -> str:
        '''
        Load the system prompt from a file.

        :param language: The language of the system prompt.
        :type language: str

        :return: The system prompt.
        :rtype: str
        '''
        lang = language.upper()

        # load prompts
        prompts_file = f'./prompts/llm_judge/{lang}.toml'
        with open(prompts_file, 'rb') as file:
            prompts = tomli.load(file)
        
        # get the system prompt
        return prompts.get('llm_judge_system')['prompt']