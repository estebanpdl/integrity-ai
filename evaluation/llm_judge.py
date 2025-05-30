# -*- coding: utf-8 -*-

'''
LLM as a Judge: Scoring Rubric
'''

# import modules
import os
import json
import time
import tomli
import string
import random
import threading

# dotenv for environment variables
from dotenv import load_dotenv

# OpenAI client
from openai import OpenAI

# Groq client
from groq import Groq
from groq.types.chat.chat_completion import ChatCompletion

# LLM as a Judge class
class LLMJudge:
    '''
    LLMJudge class
    '''

    # jitter min value
    DEFAULT_JITTER = 0.15

    def __init__(self, args: dict = None):
        '''
        Initialize the LLMJudge class.
        '''
        # get arguments
        args = args or {}
        self.args = args

        # get model name
        self.model_name = self.args.get('judge_model')
        self._model_map = self._map_models()
        if self.model_name not in self._model_map:
            raise ValueError(f'Unsupported judge model: {self.model_name}')
        
        self.evaluator_fn = self._model_map[self.model_name]

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

        # get language
        self.language = self.args.get('language', 'en')

        # load environment variables
        env_file_path = './config/.env'
        load_dotenv(env_file_path)

        # LLM clients
        self.openai_client = OpenAI()
        self.groq_client = Groq(
            api_key=os.getenv('GROQ_API_KEY')
        )
    
    def get_model_limits(self) -> dict:
        '''
        Get the model limits.

        :return: The model limits.
        :rtype: dict
        '''
        path = './config/model_limits.json'
        with open(path, 'r', encoding='utf-8') as f:
            model_limits = json.load(f)
        
        return model_limits['evaluation'][self.model_name]

    def _load_system_prompt(self) -> str:
        '''
        Load the system prompt from a file.

        :return: The system prompt.
        :rtype: str
        '''
        lang = self.language.upper()

        # load prompts
        prompts_file = f'./prompts/llm_judge/{lang}.toml'
        with open(prompts_file, 'rb') as file:
            prompts = tomli.load(file)
        
        # get the system prompt
        return prompts.get('llm_judge_system')['prompt']
    
    def _load_message_prompt(self, llm_generated_text: str = None) -> str:
        '''
        Load the message prompt from a file.

        :param llm_generated_text: The LLM generated text to be evaluated.
        :type llm_generated_text: str

        :return: The message prompt.
        :rtype: str
        '''
        lang = self.language.upper()

        # load prompts
        prompts_file = f'./prompts/llm_judge/{lang}.toml'
        with open(prompts_file, 'rb') as file:
            prompts = tomli.load(file)
        
        # get the message prompt
        message_prompt = prompts.get('llm_judge_message')['prompt']

        # substitute the LLM generated text into the message prompt
        return string.Template(message_prompt).substitute(
            llm_generated_text=llm_generated_text
        )
    
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

                # timestamps and tokens used
                self.request_timestamps[:] = [
                    t for t in self.request_timestamps if now - t < 60.0
                ]
                self.token_usage_log[:] = [
                    (t, tokens) for t, tokens in self.token_usage_log if now - t < 60.0
                ]

                if len(self.request_timestamps) >= self.max_requests_per_min:
                    if self.request_timestamps:
                        oldest_request_time = self.request_timestamps[0]
                        request_wait = 60.0 - (now - oldest_request_time)
                    else:
                        request_wait = 1.0
                    wait_time = max(wait_time, request_wait)
                
                tokens_used = sum(tokens for _, tokens in self.token_usage_log)
                
                # aggregated tokens
                agg_tokens = tokens_used + completion_tokens
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

                    # increment daily requests
                    self.daily_requests += 1

                    # add jitter to avoid thread pileup
                    jitter = self.DEFAULT_JITTER + random.uniform(0, 0.05)
                    self.stop_flag.wait(jitter)
                    break
            
            # wait_time != 0 - enforced rate-limit sleep - when over RPM/TPM
            jittered_wait = wait_time + random.uniform(0.05, 0.25)
            self.stop_flag.wait(jittered_wait)
            
    def evaluate_with_openai(self, llm_generated_text: str,
                             completion_tokens: int) -> dict:
        '''
        Evaluate the LLM generated text with OpenAI models.

        :param llm_generated_text: The LLM generated text to be evaluated.
        :type llm_generated_text: str

        :param completion_tokens: The number of completion tokens used.
        :type completion_tokens: int

        :return: The evaluation response from OpenAI.
        :rtype: dict
        '''
        # enforce rate limits
        self._enforce_rate_limits(completion_tokens)

        # make request
        while True:
            if self.daily_requests - 1 < self.max_requests_per_day:
                response = self.openai_client.chat.completions.with_raw_response.create(
                    model=self.model_name,
                    messages=[
                        {
                            'role': 'system',
                            'content': self._load_system_prompt()
                        },
                        {
                            'role': 'user',
                            'content': self._load_message_prompt(
                                llm_generated_text=llm_generated_text
                            )
                        }
                    ],
                    temperature=1.0,
                    response_format={'type': 'json_object'}
                )

                # update token usage log
                self.token_usage_log.append(
                    (time.time(), completion_tokens)
                )

                # return response
                return response.parse().choices[0].message.content
            else:
                remaining_requests = response.headers.get('x-ratelimit-remaining-requests')
                if remaining_requests > (self.max_requests_per_day - self.daily_requests):
                    with self.rate_limit_lock:
                        self.daily_requests = remaining_requests
                        continue
                else:
                    sleep_time = 60.0
                    self.stop_flag.wait(sleep_time)
                    continue
    
    def evaluate_with_groq(self, llm_generated_text: str,
                            completion_tokens: int) -> dict:
        '''
        Evaluate the LLM generated text with Groq models.
        '''
        pass
    
    def _map_models(self) -> dict:
        '''
        Map the model name to the appropriate provider.

        :return: The model name to the appropriate provider.
        :rtype: dict
        '''
        openai_models = [
            'gpt-4o',
            'gpt-4o-mini',
            'gpt-4.1',
            'gpt-4.1-mini',
            'gpt-4.1-nano',
            'o4-mini'
        ]
        
        return {
            model: self.evaluate_with_openai for model in openai_models
        }
