# -*- coding: utf-8 -*-

# import modules
import time
import random
import threading

# multithreading
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenAI
import openai
from openai import RateLimitError

# MongoDB connection
from databases import MongoDBManager

MAX_REQUESTS_PER_MIN = 250
REQUEST_INTERVAL = 60.0 / MAX_REQUESTS_PER_MIN

# Shared threading lock to manage timing
rate_limit_lock = threading.Lock()
last_request_time = [0.0]

def wait_for_slot():
    '''
    '''
    with rate_limit_lock:
        now = time.time()
        elapsed = now - last_request_time[0]
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        
        last_request_time[0] = time.time()


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

def process_multiple(prompts):
    '''
    '''
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_prompt = {executor.submit(chat_with_backoff_threadsafe, p): p for p in prompts}
        for future in as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f'{prompt} generated an exception: {exc}')





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

print (uuids)
