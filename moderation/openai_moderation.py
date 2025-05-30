# -*- coding: utf-8 -*-

# import modules
import json
import time
import random
import threading

# dotenv for environment variables
from dotenv import load_dotenv

# OpenAI related modules
from openai import OpenAI

# OpenAIModeration class
class OpenAIModeration:
    '''
    OpenAIModeration class
    '''

    # jitter min value
    DEFAULT_JITTER = 0.15

    def __init__(self):
        '''
        Initialize the OpenAIModeration class.
        '''
        # OpenAI moderation model
        self.model = 'omni-moderation-latest'

        # get model limits
        self.model_limits = self.get_model_limits()
        self.max_requests_per_min = self.model_limits['rpm']
        self.max_tokens_per_min = self.model_limits['tpm']
        self.max_requests_per_day = self.model_limits['rpd']

        # shared threading locks
        self.rate_limit_lock = threading.Lock()
        self.token_lock = threading.Lock()

        # shared controls for rate limiting
        self.request_timestamps = []
        self.token_usage_log = []
        self.daily_requests = 0

        # shared threading event
        self.stop_flag = threading.Event()

        # load environment variables
        env_file_path = './config/.env'
        load_dotenv(env_file_path)

        # OpenAI client
        self.client = OpenAI()

    def get_model_limits(self) -> dict:
        '''
        Get the model limits.

        :return: The model limits.
        :rtype: dict
        '''
        path = './config/model_limits.json'
        with open(path, 'r', encoding='utf-8') as f:
            model_limits = json.load(f)
        
        return model_limits['openai_moderation']['omni-moderation-latest']
    
    def temp_log(self, message: str) -> None:
        '''
        Temporary log.
        '''
        path = './logs/openai_moderation.log'
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f'{message}\n')
    
    def _enforce_rate_limits(self, completion_tokens: int) -> None:
        '''
        Enforce rate limits for the OpenAI API.
        This method ensures that the number of requests and tokens used
        per minute does not exceed the specified limits.

        :param completion_tokens: The number of completion tokens used.
        :type completion_tokens: int

        :return: None
        :rtype: None
        '''
        while True:
            wait_time = 0
            with self.rate_limit_lock, self.token_lock:
                now = time.time()

                self.temp_log(f'\n\n')

                # timestamps and tokens used
                self.request_timestamps[:] = [
                    t for t in self.request_timestamps if now - t < 60.0
                ]
                self.temp_log(f'request_timestamps: {self.request_timestamps}')
                self.token_usage_log[:] = [
                    (t, tokens) for t, tokens in self.token_usage_log if now - t < 60.0
                ]
                self.temp_log(f'token_usage_log: {self.token_usage_log}')

                if len(self.request_timestamps) >= self.max_requests_per_min:
                    if self.request_timestamps:
                        oldest_request_time = self.request_timestamps[0]
                        request_wait = 60.0 - (now - oldest_request_time)
                    else:
                        request_wait = 1.0
                    wait_time = max(wait_time, request_wait)
                
                tokens_used = sum(tokens for _, tokens in self.token_usage_log)
                self.temp_log(f'tokens_used: {tokens_used}')

                # aggregated tokens
                agg_tokens = tokens_used + completion_tokens
                self.temp_log(f'agg_tokens: {agg_tokens}')
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
                    self.temp_log(f'request_timestamps added: {self.request_timestamps}')

                    # increment daily requests
                    self.daily_requests += 1
                    self.temp_log(f'daily_requests: {self.daily_requests}')

                    # add jitter to avoid thread pileup
                    jitter = self.DEFAULT_JITTER + random.uniform(0, 0.05)
                    self.stop_flag.wait(jitter)
                    break
            
            # wait_time != 0 - enforced rate-limit sleep - when over RPM/TPM
            jittered_wait = wait_time + random.uniform(0.05, 0.25)
            self.stop_flag.wait(jittered_wait)
    
    def audit_generated_content(self,
                                completion_tokens: int,
                                content: str) -> dict:
        '''
        Audit the generated content by the LLM.

        :param completion_tokens: The number of completion tokens used.
        :type completion_tokens: int

        :param content: The content to be audited.
        :type content: str

        :return: The audit result.
        :rtype: dict
        '''
        # enforce rate limits
        self._enforce_rate_limits(completion_tokens)

        # make request
        if self.daily_requests < self.max_requests_per_day:
            response = self.client.moderations.create(
                model=self.model,
                input=content
            )

            # update token usage log
            self.token_usage_log.append(
                (time.time(), completion_tokens)
            )

            # return response
            return response.model_dump()
        else:
            raise Exception('Daily request limit reached')
