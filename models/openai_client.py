# -*- coding: utf-8 -*-

# import modules
import os
import json
import time
import random
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
    # OpenAI model limits
    MODEL_LIMITS = {
        'gpt-4o-mini': {
            'rpm': 500,
            'tpm': 200000
        }
    }

    # maximum number of concurrent requests
    MAX_CONCURRENT_REQUESTS = 10

    # jitter min value
    DEFAULT_JITTER = 0.25

    def __init__(self, model_name: str):
        '''
        Initialize the OpenAIGPT class.

        :param model_name: The name of the OpenAI model to be used.
        '''
        super().__init__()

        # load environment variables
        env_file_path = './config/.env'
        load_dotenv(env_file_path)

        # OpenAI client
        self.client = OpenAI()

        # OpenAI model
        self.model_name = model_name

        # get model limits
        self.max_requests_per_min = self.MODEL_LIMITS[self.model_name]['rpm']
        self.max_tokens_per_min = self.MODEL_LIMITS[self.model_name]['tpm']

        # model token encoding
        self.encoding = tiktoken.encoding_for_model(self.model_name)

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

    def estimated_tokens(self, prompt: str) -> int:
        '''
        Estimate the number of tokens in a prompt.

        :param prompt: The prompt to be estimated.
        :return: The estimated number of tokens.
        '''

        # estimate tokens
        return len(self.encoding.encode(prompt))

    def _get_average_response_tokens(self) -> int:
        '''
        Get the average number of tokens used in responses.

        :return: The average number of tokens used in responses.
        '''
        if not self.token_usage_log:
            return 300
        
        recent = self.token_usage_log[-10:]
        return int(sum(n for _, n in recent) / len(recent))

    def _enforce_rate_limits(self, estimated_tokens: int = 0) -> None:
        '''
        Enforce rate limits for the OpenAI API.
        This method ensures that the number of requests and tokens used
        per minute does not exceed the specified limits.

        :param estimated_tokens: The estimated tokens in the prompt.
        '''
        # wait for rate limit slot
        while True:
            wait_time = 0.0
            with self.rate_limit_lock, self.token_lock:
                now = time.time()

                # timestamps and tokens used
                self.request_timestamps[:] = [
                    t for t in self.request_timestamps if now - t < 60.0
                ]
                self.token_usage_log[:] = [
                    (t, n) for t, n in self.token_usage_log if now - t < 60.0
                ]

                if len(self.request_timestamps) >= self.max_requests_per_min:
                    if self.request_timestamps:
                        oldest_request_time = self.request_timestamps[0]
                        request_wait = 60.0 - (now - oldest_request_time)
                    else:
                        request_wait = 1.0
                    wait_time = max(wait_time, request_wait)
                
                tokens_used = sum(n for _, n in self.token_usage_log)

                # response buffer
                response_buffer = self._get_average_response_tokens()
                agg_tokens = tokens_used + estimated_tokens + response_buffer
                if agg_tokens > self.max_tokens_per_min:
                    if self.token_usage_log:
                        oldest_token_time = self.token_usage_log[0][0]
                        token_wait = 60.0 - (now - oldest_token_time)
                    else:
                        token_wait = 1.0
                    wait_time = max(wait_time, token_wait)
                
                if wait_time == 0.0:
                    # no enforced wait time
                    now = time.time()
                    self.request_timestamps.append(now)
                    self.token_usage_log.append((now, estimated_tokens))

                    # add jitter to avoid thread pileup
                    jitter = self.DEFAULT_JITTER + random.uniform(0, 0.05)
                    time.sleep(jitter)

            # enforced rate-limit sleep - when over RPM/TPM
            jittered_wait = wait_time + random.uniform(0.05, 0.25)
            time.sleep(jittered_wait)
    
    def _call_with_backoff(self, prompt: str, max_retries: int = 5,
                           semaphore: threading.Semaphore = None) -> str:
        '''
        Generate content with backoff strategy for rate limiting.

        :param prompt: The prompt to be generated.
        :param max_retries: The maximum number of retries in case of rate
            limiting. Default is 5.
        
        :return: The generated content.
        '''
        while not self.stop_flag.is_set():
            with self.semaphore:
                try:
                    # estimate tokens
                    estimated_tokens = self.estimated_tokens(prompt)

                    # enforce rate limits
                    self._enforce_rate_limits(estimated_tokens)

                    # generate content
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                'role': 'user',
                                'content': prompt
                            }
                        ]
                    )
                    return response['choices'][0]['message']['content']
                except Exception as e:
                    if isinstance(e, RateLimitError):
                        # handle rate limit error
                        print('Rate limit hit. Retrying...')
                        time.sleep(2 ** max_retries + random.uniform(0, 1))
                        continue
                    else:
                        raise e
                finally:
                    # release semaphore
                    if semaphore is not None:
                        semaphore.release()
        return None

    def test_completions(self, user_prompt: str = None):
        '''
        '''
        # make request
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    'role': 'user',
                    'content': user_prompt
                }
            ]
        )

        return response
