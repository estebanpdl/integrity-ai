# -*- coding: utf-8 -*-

# import modules
import os
import ast
import json
import time
import signal
import random
import tiktoken
import threading

# import base class
from .model import LanguageModel

# OpenAI related modules
import openai
from openai import OpenAI, RateLimitError

# multithreading
from concurrent.futures import ThreadPoolExecutor, as_completed

# MongoDB connection
from databases import MongoDBManager

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
        },
        'gpt-4.1-mini': {
            'rpm': 500,
            'tpm': 200000
        },
        'gpt-4.1-nano': {
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
        try:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        except Exception as KeyError:
            self.encoding = tiktoken.get_encoding('o200k_base')

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

        :param prompt: The prompt to be estimated.
        :type prompt: str
        
        :return: The estimated number of tokens.
        :rtype: int
        '''
        # estimate tokens
        return len(self.encoding.encode(prompt))
    
    def _enforce_rate_limits(self, estimated_tokens: int) -> None:
        '''
        Enforce rate limits for the OpenAI API.
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
    
    def _process_response(self, response: openai.ChatCompletion) -> None:
        '''
        Process API response.

        :param response: The response from the OpenAI API.
        :type response: openai.ChatCompletion

        :return: None
        '''
        # get tokens used
        prompt_tokens_used = response.usage.prompt_tokens
        completion_tokens_used = response.usage.completion_tokens

        # update token usage log
        self.token_usage_log.append(
            (
                time.time(),
                prompt_tokens_used,
                completion_tokens_used
            )
        )

        # get response
        response_content = response.choices[0].message.content

        # get collection
        collection = self.mongodb_manager.get_collection(
            db_name='narrative-blueprint',
            collection_name='gpt-4o-mini'
        )

        # parse response
        response_content = ast.literal_eval(response_content)

        # upload response to database
        collection.insert_one(
            response_content
        )
    
    def _call_with_backoff(self, message: str, request_id: int,
                           max_retries: int = 5) -> None:
        '''
        '''
        retry_count = 0
        while not self.stop_flag.is_set() and retry_count <= max_retries:
            with self.semaphore:
                try:
                    # estimate tokens
                    system_prompt = message[0]['content']
                    user_prompt = message[1]['content']

                    prompt = f'{system_prompt}\n{user_prompt}'
                    estimated_tokens = self._estimate_tokens(prompt)
                    
                    # enforce rate limits
                    self._enforce_rate_limits(estimated_tokens)

                    # make request
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=message
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
    
    def _test_messages(self, message: str, request_id: int) -> int:
        '''
        Test the messages.

        :param message: The message to be tested.
        :type message: list

        :param request_id: The request ID.
        :type request_id: int

        :return: The estimated tokens.
        :rtype: int
        '''
        system_prompt = message[0]['content']
        user_prompt = message[1]['content']

        prompt = f'{system_prompt}\n{user_prompt}'
        estimated_tokens = self._estimate_tokens(prompt)

        return estimated_tokens
    
    def test_prompts(self, message: list) -> str:
        '''
        '''
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message
        )

        return response












    # def run_parallel_prompt_tasks(self, messages: list) -> None:
    #     '''
    #     Run parallel prompt tasks.

    #     :param messages: The list of messages to be processed.
    #     '''
    #     signal.signal(signal.SIGINT, self._signal_handler)

    #     results = []
    #     with ThreadPoolExecutor(max_workers=30) as executor:
    #         futures = {
    #             executor.submit(self._call_with_backoff, message, i): i
    #             for i, message in enumerate(messages)
    #         }

    #         for future in as_completed(futures):
    #             try:
    #                 result = future.result()
    #                 if result:
    #                     results.append(result)
    #             except Exception as e:
    #                 print(f"[MAIN] Exception from future: {e}")

    #     print("\n[SUMMARY] All requests completed.")
    #     print(f"Total results: {len(results)}")

    # def chat_with_backoff(self, message, request_id: int) -> None:
    #     '''
    #     '''
    #     while not self.stop_flag.is_set():
    #         with self.semaphore:
    #             try:
    #                 print(f"[REQUEST NUMBER] {request_id}")
    #                 estimated_tokens = self._estimate_tokens(message)
    #                 print(f"[INFO] Estimated tokens: {estimated_tokens}")

    #                 self._enforce_rate_limits(estimated_tokens)

    #                 print(f"[CALL] Proceeding with request at {time.strftime('%X')}")
    #                 actual_tokens_used = estimated_tokens + random.randint(-10, 10)
    #                 print(f"[SIMULATION] Tokens used (real): {actual_tokens_used}")

    #                 with self.token_lock:
    #                     self.token_usage_log.append((time.time(), actual_tokens_used))

    #                 return {"message": "Simulated response", "tokens": actual_tokens_used}
    #             except Exception as e:
    #                 print(f"[ERROR] Request {request_id} failed: {e}")

    # def _enforce_rate_limits(self, estimated_tokens: int) -> None:
    #     '''
    #     Enforce rate limits for the OpenAI API.
    #     This method ensures that the number of requests and tokens used
    #     per minute does not exceed the specified limits.

    #     :param estimated_tokens: The estimated tokens in the prompt.
    #     '''
    #     # wait for rate limit slot
    #     while True:
    #         wait_time = 0
    #         with self.rate_limit_lock, self.token_lock:
    #             now = time.time()

    #             # timestamps and tokens used
    #             self.request_timestamps[:] = [
    #                 t for t in self.request_timestamps if now - t < 60.0
    #             ]
    #             self.token_usage_log[:] = [
    #                 (t, n) for t, n in self.token_usage_log if now - t < 60.0
    #             ]

    #             if len(self.request_timestamps) >= self.max_requests_per_min:
    #                 if self.request_timestamps:
    #                     oldest_request_time = self.request_timestamps[0]
    #                     request_wait = 60.0 - (now - oldest_request_time)
    #                 else:
    #                     request_wait = 1.0
    #                 wait_time = max(wait_time, request_wait)
                
    #             tokens_used = sum(n for _, n in self.token_usage_log)

    #             print(f'[REQUESTS TIMESTAMPS]: {len(self.request_timestamps)}')
    #             print(f'[TOKENS USAGE LOG]: {len(self.token_usage_log)}')
    #             print(f"[TOKENS USED] {tokens_used}")

    #             # response buffer
    #             response_buffer = self._get_average_response_tokens()
    #             agg_tokens = tokens_used + estimated_tokens + response_buffer
    #             if agg_tokens > self.max_tokens_per_min:
    #                 if self.token_usage_log:
    #                     oldest_token_time = self.token_usage_log[0][0]
    #                     token_wait = 60.0 - (now - oldest_token_time)
    #                 else:
    #                     token_wait = 1.0
    #                 wait_time = max(wait_time, token_wait)
                
    #             if wait_time == 0:
    #                 # no enforced wait time
    #                 now = time.time()
    #                 self.request_timestamps.append(now)
    #                 self.token_usage_log.append((now, estimated_tokens))

    #                 # add jitter to avoid thread pileup
    #                 jitter = self.DEFAULT_JITTER + random.uniform(0, 0.05)
    #                 print(f"[JITTERED WAIT] Sleeping {jitter:.2f}s to avoid thread clustering")
    #                 time.sleep(jitter)
    #                 break

    #         # enforced rate-limit sleep - when over RPM/TPM
    #         jittered_wait = wait_time + random.uniform(0.05, 0.25)
    #         print(f"\n\n[WAIT] Sleeping {jittered_wait:.2f}s (RPM/TPM limit)\n\n")
    #         time.sleep(jittered_wait)

    # def _chat_with_backoff_threadsafe(self, prompt, max_retries=5):
    #     '''
    #     Chat with the LLM engine, implementing a backoff strategy for rate
    #     limiting.
        
    #     :param prompt: The prompt to be sent to the LLM engine.
    #     :param max_retries: The maximum number of retries in case of rate
    #         limiting. Default is 5.
    #     '''
    #     retry_count = 0
    #     while retry_count <= max_retries:
    #         try:
    #             self._wait_for_slot()
    #             response = self.llm_engine.chat(prompt)
    #             return response
    #         except Exception as e:
    #             retry_count += 1
    #             sleep_time = (2 ** retry_count) + random.uniform(0, 1)
    #             print ('Rate limit hit. Retrying in {sleep_time} seconds...')
    #             time.sleep(sleep_time)
        
    #     return 'Failed after retries'
    
    # def chat_with_backoff_threadsafe_bis(self, prompt, max_retries=5):
    #     '''
    #     '''
    #     retry_count = 0
    #     while retry_count <= max_retries:
    #         try:
    #             self.wait_for_slot()
    #             response = prompt
    #             return response
    #         except RateLimitError:
    #             retry_count += 1
    #             sleep_time = (2 ** retry_count) + random.uniform(0, 1)
    #             print ('Rate limit hit. Retrying in {sleep_time} seconds...')
    #             time.sleep(sleep_time)
        
    #     return 0

    # def _call_with_backoff(self, prompt: str, max_retries: int = 5) -> str:
    #     '''
    #     Generate content with backoff strategy for rate limiting.

    #     :param prompt: The prompt to be generated.
    #     :param max_retries: The maximum number of retries in case of rate
    #         limiting. Default is 5.
        
    #     :return: The generated content.
    #     '''
    #     while not self.stop_flag.is_set():
    #         with self.semaphore:
    #             try:
    #                 # estimate tokens
    #                 estimated_tokens = self.estimate_tokens(prompt)

    #                 # enforce rate limits
    #                 self._enforce_rate_limits(estimated_tokens)

    #                 # generate content
    #                 response = self.client.chat.completions.create(
    #                     model=self.model_name,
    #                     messages=[
    #                         {
    #                             'role': 'user',
    #                             'content': prompt
    #                         }
    #                     ]
    #                 )
    #                 return response['choices'][0]['message']['content']
    #             except Exception as e:
    #                 if isinstance(e, RateLimitError):
    #                     # handle rate limit error
    #                     print('Rate limit hit. Retrying...')
    #                     time.sleep(2 ** max_retries + random.uniform(0, 1))
    #                     continue
    #                 else:
    #                     raise e
    #             finally:
    #                 # release semaphore
    #                 if self.semaphore is not None:
    #                     self.semaphore.release()
        
    #     return None
    

