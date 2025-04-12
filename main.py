# -*- coding: utf-8 -*-

# import modules
import time
import random

# OpenAI
from openai import RateLimitError

# MongoDB connection
from databases import MongoDBManager

def wait_for_slot():
    '''
    '''
    pass

def chat_with_backoff_threadsafe(prompt, max_retries=5):
    '''
    '''
    retry_count = 0
    while retry_count <= max_retries:
        try:
            wait_for_slot()
            response = prompt
            return response
        except RateLimitError:
            retry_count += 1
            sleep_time = (2 ** retry_count) + random.uniform(0, 1)
            print ('Rate limit hit. Retrying in {sleep_time} seconds...')
            time.sleep(sleep_time)
    
    return 0

# MongoDBManager class
mongodb_manager = MongoDBManager()
collection = mongodb_manager.create_connection(
    'narrative-blueprint',
    'gpt-4o-mini'
)

sample_document = { "uuid": "f0aabce3-5e06-4754-a66a-a4a9c69b75bb", "type": "sample" }
collection.insert_one(sample_document)

uuids = mongodb_manager.get_collected_uuids(
    'narrative-blueprint',
    'gpt-4o-mini'
)
