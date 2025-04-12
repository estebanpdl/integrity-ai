# -*- coding: utf-8 -*-

'''
Narrative blueprint: An structured analysis of a piece of real-world
disinformation. It's a data-rich description of the manipulative techniques
used. It provides the raw material that informs the creation of the evaluation
prompt or test case.

This module will handle the processing of narrative blueprints and their
integration into the evaluation framework.

'''

# import modules
import json
import time
import tomli
import string
import random
import threading

# multithreading
from concurrent.futures import ThreadPoolExecutor, as_completed

# import LLM base class
from models import LanguageModel

# MongoDB connection
from databases import MongoDBManager

# Narrative blueprint class
class NarrativeBlueprint:
    '''
    Narrative blueprint class
    '''
    def __init__(self, llm_engine: LanguageModel):
        '''
        Initialize the NarrativeBlueprint class.
        
        :param llm_engine: The LLM engine to be used for narrative blueprint
            processing
        '''
        # LLM engine
        self.llm_engine = llm_engine
    
    '''
    load dataset
    load prompts
    
    '''
    
    def _chat_with_backoff_threadsafe(self, prompt, max_retries=5):
        '''
        Chat with the LLM engine, implementing a backoff strategy for rate
        limiting.
        
        :param prompt: The prompt to be sent to the LLM engine.
        :param max_retries: The maximum number of retries in case of rate
            limiting. Default is 5.
        '''
        retry_count = 0
        while retry_count <= max_retries:
            try:
                self._wait_for_slot()
                response = self.llm_engine.chat(prompt)
                return response
            except Exception as e:
                retry_count += 1
                sleep_time = (2 ** retry_count) + random.uniform(0, 1)
                print ('Rate limit hit. Retrying in {sleep_time} seconds...')
                time.sleep(sleep_time)
        
        return 'Failed after retries'
    