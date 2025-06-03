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
        self.judge_max_requests_per_min = self.model_limits['rpm']
        self.judge_max_tokens_per_min = self.model_limits['tpm']
        self.judge_max_requests_per_day = self.model_limits['rpd']

        # shared threading locks
        self.judge_rate_limit_lock = threading.Lock()
        self.judge_token_lock = threading.Lock()

        # shared controls for rate limiting
        self.judge_request_timestamps = []
        self.judge_token_usage_log = []
        self.judge_daily_requests = 0

        # shared threading event
        self.stop_flag = threading.Event()

        # get language
        self.language = self.args.get('language', 'en')

        # load environment variables
        env_file_path = './config/.env'
        load_dotenv(env_file_path)

        # LLM clients
        self.openai_client = OpenAI()
    
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
    
    def _judge_enforce_rate_limits(self, completion_tokens: int) -> None:
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
            with self.judge_rate_limit_lock, self.judge_token_lock:
                now = time.time()

                # timestamps and tokens used
                self.judge_request_timestamps[:] = [
                    t for t in self.judge_request_timestamps if now - t < 60.0
                ]
                self.judge_token_usage_log[:] = [
                    (t, tokens) for t, tokens in self.judge_token_usage_log if now - t < 60.0
                ]

                if len(self.judge_request_timestamps) >= self.judge_max_requests_per_min:
                    if self.judge_request_timestamps:
                        oldest_request_time = self.judge_request_timestamps[0]
                        request_wait = 60.0 - (now - oldest_request_time)
                    else:
                        request_wait = 1.0
                    wait_time = max(wait_time, request_wait)
                
                tokens_used = sum(tokens for _, tokens in self.judge_token_usage_log)
                
                # aggregated tokens
                agg_tokens = tokens_used + completion_tokens
                if agg_tokens > self.judge_max_tokens_per_min:
                    if self.judge_token_usage_log:
                        oldest_token_time = self.judge_token_usage_log[0][0]
                        token_wait = 60.0 - (now - oldest_token_time)
                    else:
                        token_wait = 1.0
                    wait_time = max(wait_time, token_wait)
                
                if wait_time == 0:
                    # no enforced wait time
                    now = time.time()
                    self.judge_request_timestamps.append(now)

                    # increment daily requests
                    self.judge_daily_requests += 1

                    # add jitter to avoid thread pileup
                    jitter = self.DEFAULT_JITTER + random.uniform(0, 0.05)
                    self.stop_flag.wait(jitter)
                    break
            
            # wait_time != 0 - enforced rate-limit sleep - when over RPM/TPM
            jittered_wait = wait_time + random.uniform(0.05, 0.25)
            self.stop_flag.wait(jittered_wait)
            
    def evaluate_with_openai(self, completion_tokens: int,
                             llm_generated_text: str) -> dict:
        '''
        Evaluate the LLM generated text with OpenAI models.

        :param completion_tokens: The number of completion tokens used.
        :type completion_tokens: int

        :param llm_generated_text: The LLM generated text to be evaluated.
        :type llm_generated_text: str

        :return: The evaluation response from OpenAI.
        :rtype: dict
        '''
        # check if test mode is enabled
        if self.args.get('test', False):
            # return dummy response for testing
            return {
                'model': self.model_name,
                'results': {
                    'score': 0,
                    'reason': 'Not implemented - test mode'
                }
            }

        # enforce rate limits
        self._judge_enforce_rate_limits(completion_tokens)

        # make request
        max_retries = 3
        retry_count = 0
        while retry_count <= max_retries:
            if self.judge_daily_requests - 1 < self.judge_max_requests_per_day:
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
                    temperature=0.5,
                    response_format={'type': 'json_object'}
                )

                # update token usage log
                self.judge_token_usage_log.append(
                    (time.time(), completion_tokens)
                )

                # parse response
                response_content = json.loads(
                    response.parse().choices[0].message.content
                )

                # return response
                return {
                    'model': self.model_name,
                    'results': response_content
                }
            else:
                response = None
                remaining_requests = response.headers.get('x-ratelimit-remaining-requests')
                if remaining_requests > (self.judge_max_requests_per_day - self.judge_daily_requests):
                    with self.judge_rate_limit_lock:
                        self.judge_daily_requests = remaining_requests
                        continue
                else:
                    retry_count += 1
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

        anthropic_models = [
            'claude-3-7-sonnet-latest',
            'claude-sonnet-4-20250514',
            'claude-opus-4-20250514'
        ]

        google_models = [    
            'gemini-2.5-flash-preview-05-20',
            'gemini-2.5-pro-preview-05-06'
        ]
        
        return {
            model: self.evaluate_with_openai for model in openai_models
        }
