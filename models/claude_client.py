# -*- coding: utf-8 -*-

# import modules
import os
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
        self.max_requests_per_min = self.MODEL_LIMITS[self.model_name]['rpm']
        self.max_tokens_per_min = self.MODEL_LIMITS[self.model_name]['tpm']

        # API limits
        self.request_interval = 60.0 / self.max_requests_per_min

        # shared threading locks
        self.rate_limit_lock = threading.Lock()
        self.token_lock = threading.Lock()
        self.stop_flag = threading.Event()

        # share controls for rate limiting
        self.request_timestamps = []
        self.last_request_time = [0.0]
        self.token_usage_log = []

        # threading semaphore
        self.semaphore = threading.Semaphore(self.MAX_CONCURRENT_REQUESTS)

        # MongoDB connection
        self.mongodb_manager = MongoDBManager()

    def _get_average_completion_tokens(self) -> int:
        '''
        Get the average number of tokens used in responses.

        :return: The average number of tokens used in responses.
        :rtype: int
        '''
        if not self.token_usage_log:
            return self.AVERAGE_TOKEN_USAGE
        
        completion_tokens = [c for _, _, c in self.token_usage_log]
        return int(sum(completion_tokens) / len(completion_tokens))

    def _estimate_tokens(self, prompt: str) -> int:
        '''
        Estimate the number of tokens in a prompt.
        Claude API doesn't have a direct token counting tool like tiktoken,
        so we use a rough approximation of 4 characters per token.

        :param prompt: The prompt to be estimated.
        :type prompt: str
        
        :return: The estimated number of tokens.
        :rtype: int
        '''
        # Anthropic's rough approximation is ~4 chars per token
        return len(prompt) // 4
    
    def _enforce_rate_limits(self, estimated_tokens: int) -> None:
        '''
        Enforce rate limits for the Anthropic API.
        This method ensures that the number of requests and tokens used
        per minute does not exceed the specified limits.

        :param estimated_tokens: The estimated tokens in the prompt.
        :type estimated_tokens: int

        :return: None
        '''
        # wait for rate limit slot
        while True:
            wait_time = 0
            with self.rate_limit_lock, self.token_lock:
                now = time.time()

                # timestamps and tokens used
                self.request_timestamps[:] = [
                    t for t in self.request_timestamps if now - t < 60.0
                ]
                self.token_usage_log[:] = [
                    (t, p, c) for t, p, c in self.token_usage_log if now - t < 60.0
                ]

                if len(self.request_timestamps) >= self.max_requests_per_min:
                    if self.request_timestamps:
                        oldest_request_time = self.request_timestamps[0]
                        request_wait = 60.0 - (now - oldest_request_time)
                    else:
                        request_wait = 1.0
                    wait_time = max(wait_time, request_wait)
                
                tokens_used = sum(p + c for _, p, c in self.token_usage_log)

                # response buffer
                response_buffer = self._get_average_completion_tokens()
                agg_tokens = tokens_used + estimated_tokens + response_buffer
                if agg_tokens > self.max_tokens_per_min:
                    if self.token_usage_log:
                        oldest_token_time = self.token_usage_log[0][0]
                        token_wait = 60.0 - (now - oldest_token_time)
                    else:
                        token_wait = 1.0
                    wait_time = max(wait_time, token_wait)
                
                if wait_time == 0:
                    # no enforced wait time
                    now = time.time()
                    self.request_timestamps.append(now)

                    # add jitter to avoid thread pileup
                    jitter = self.DEFAULT_JITTER + random.uniform(0, 0.05)
                    time.sleep(jitter)
                    break

            # enforced rate-limit sleep - when over RPM/TPM
            jittered_wait = wait_time + random.uniform(0.05, 0.25)
            time.sleep(jittered_wait)
    
    def _signal_handler(self, sig, frame) -> None:
        '''
        Signal handler for Ctrl+C.
        '''
        self.stop_flag.set()
    
    def _process_response(self, response) -> None:
        '''
        Process API response.

        :param response: The response from the Anthropic API.
        :type response: anthropic.types.message.Message

        :return: None
        '''
        # get tokens used
        prompt_tokens_used = response.usage.input_tokens
        completion_tokens_used = response.usage.output_tokens

        # update token usage log
        self.token_usage_log.append(
            (
                time.time(),
                prompt_tokens_used,
                completion_tokens_used
            )
        )

        # get response
        response_content = response.content[0].text

        # get collection
        collection = self.mongodb_manager.get_collection(
            db_name='narrative-blueprint',
            collection_name=self.model_name
        )

        # parse response
        try:
            response_content = ast.literal_eval(response_content)
        except (SyntaxError, ValueError):
            # Handle case where the response is not valid Python literal
            response_content = {"raw_content": response_content}

        # upload response to database
        collection.insert_one(
            response_content
        )
    
    def _call_with_backoff(self, message: list, request_id: int,
                           max_retries: int = 5) -> None:
        '''
        Make API call with exponential backoff retry strategy.
        
        :param message: The message to be sent.
        :type message: list
        
        :param request_id: The ID of the request.
        :type request_id: int
        
        :param max_retries: Maximum number of retries.
        :type max_retries: int
        
        :return: True if successful, None otherwise.
        '''
        retry_count = 0
        while not self.stop_flag.is_set() and retry_count <= max_retries:
            with self.semaphore:
                try:
                    # Convert OpenAI format to Anthropic format
                    system_prompt = None
                    user_content = None
                    
                    for msg in message:
                        if msg['role'] == 'system':
                            system_prompt = msg['content']
                        elif msg['role'] == 'user':
                            user_content = msg['content']
                    
                    # If no system prompt is provided, use empty string
                    if system_prompt is None:
                        system_prompt = ""
                    
                    # If no user content is provided, use empty string
                    if user_content is None:
                        user_content = ""
                        
                    prompt = f'{system_prompt}\n{user_content}'
                    estimated_tokens = self._estimate_tokens(prompt)
                    
                    # enforce rate limits
                    self._enforce_rate_limits(estimated_tokens)

                    # make request
                    response = self.client.messages.create(
                        model=self.model_name,
                        system=system_prompt,
                        messages=[
                            {
                                "role": "user",
                                "content": user_content
                            }
                        ],
                        max_tokens=4096
                    )

                    self._process_response(response)
                    return True
                    
                except RateLimitError:
                    retry_count += 1
                    sleep_time = (2 ** retry_count) + random.uniform(0, self.DEFAULT_JITTER)

                    # sleep
                    time.sleep(sleep_time)

    def run_parallel_prompt_tasks(self, messages: list) -> list:
        '''
        Run parallel prompt tasks.

        :param messages: The list of messages to be processed.
        :type messages: list

        :return: The list of results.
        '''
        signal.signal(signal.SIGINT, self._signal_handler)

        results = []
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = {
                executor.submit(self._call_with_backoff, message, i): i
                for i, message in enumerate(messages)
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    pass
        
        return results
    
    def test_prompts(self, message: list) -> str:
        '''
        Test a prompt with the Claude API.
        
        :param message: The message to test.
        :type message: list
        
        :return: The API response.
        '''
        # Convert OpenAI format to Anthropic format
        system_prompt = None
        user_content = None
        
        for msg in message:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            elif msg['role'] == 'user':
                user_content = msg['content']
        
        # If no system prompt is provided, use empty string
        if system_prompt is None:
            system_prompt = ""
        
        # If no user content is provided, use empty string
        if user_content is None:
            user_content = ""
            
        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            max_tokens=4096
        )

        return response
