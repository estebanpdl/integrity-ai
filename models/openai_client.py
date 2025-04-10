# -*- coding: utf-8 -*-

# import modules
import os
import json
import tiktoken
import threading

# import base class
from .model import LanguageModel

# OpenAI related modules
from openai import OpenAI, RateLimitError

# dotenv for environment variables
from dotenv import load_dotenv

# OpenAIGPT class
class OpenAIGPT(LanguageModel):
    '''
    OpenAIGPT class
    '''
    def __init__(self):
        '''
        Initialize the OpenAIGPT class.
        '''
        super().__init__()

        # load environment variables
        env_file_path = './config/.env'
        load_dotenv(env_file_path)

        # OpenAI client
        self.client = OpenAI()

        # API limits
        self.max_requests_per_min = 500
        self.max_tokens_per_min = 200000
        self.request_interval = 60.0 / self.max_requests_per_min

        # shared controls
        self.rate_limit_lock = threading.Lock()
        self.token_lock = threading.Lock()
        self.last_request_time = [0.0]

    def _wait_for_slot(self):
        '''
        Wait for a slot to be available for processing requests.
        '''
        pass

    def _estimated_tokens(self, prompt: str) -> int:
        '''
        Estimate the number of tokens in a prompt.

        :param prompt: The prompt to be estimated.
        :return: The estimated number of tokens.
        '''
        # load encoding
        encoding = tiktoken.encoding_for_model(self.model_name)

        # estimate tokens
        return len(encoding.encode(prompt))
