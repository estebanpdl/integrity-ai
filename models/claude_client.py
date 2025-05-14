# -*- coding: utf-8 -*-

# import modules
import ast
import json
import time
import signal
import random
import threading

# import base class
from .model import LanguageModel

# Anthropic related modules
import anthropic
from anthropic import Anthropic, RateLimitError

# multithreading
from concurrent.futures import ThreadPoolExecutor, as_completed

# MongoDB connection
from databases import MongoDBManager

# dotenv for environment variables
from dotenv import load_dotenv

# ClaudeAI class
class ClaudeAI(LanguageModel):
    '''
    ClaudeAI class
    '''
    # Anthropic model limits
    MODEL_LIMITS = {
        'claude-3-opus-20240229': {
            'rpm': 100,
            'tpm': 100000
        },
        'claude-3-sonnet-20240229': {
            'rpm': 250, 
            'tpm': 150000
        },
        'claude-3-haiku-20240307': {
            'rpm': 500,
            'tpm': 200000
        }
    }

    # average token usage
    AVERAGE_TOKEN_USAGE = 1000

    # maximum number of concurrent requests
    MAX_CONCURRENT_REQUESTS = 10

    # jitter min value
    DEFAULT_JITTER = 0.15

    def __init__(self, model_name: str):
        '''
        Initialize the ClaudeAI class.

        :param model_name: The name of the Anthropic model to be used.
        '''
        super().__init__()

        # load environment variables
        env_file_path = './config/.env'
        load_dotenv(env_file_path)

        # Anthropic client
        self.client = Anthropic()

        # Anthropic model
        self.model_name = model_name

        # get model limits
        pass
