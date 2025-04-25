# -*- coding: utf-8 -*-

# import modules
import os
import ast
import time
import signal
import random
import openai
import tiktoken
import threading

# dotenv for environment variables
from dotenv import load_dotenv

# multithreading
from concurrent.futures import ThreadPoolExecutor, wait

# tqdm bar
from tqdm import tqdm

# import base class
from .model import LanguageModel

# OpenAI related modules
from openai import OpenAI, RateLimitError

# MongoDB connection
from databases import MongoDBManager

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

        # log file
        self.log_file = './logs/openai_client.log'

        # API limits
        self.request_interval = 60.0 / self.max_requests_per_min

        # shared threading locks
        self.rate_limit_lock = threading.Lock()
        self.token_lock = threading.Lock()
        self.tqdm_lock = threading.Lock()

        # shared threading event
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
    
    def _log_write(self, message: str) -> None:
        '''
        Write a message to the log file.

        :param message: The message to be written to the log file.
        :type message: str

        :return: None
        '''
        # create log directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # write to log file
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def _signal_handler(self, sig, frame) -> None:
        '''
        Signal handler for Ctrl+C.

        :param sig: The signal number.
        :type sig: int

        :param frame: The current stack frame.
        :type frame: frame

        :return: None
        '''
        self.stop_flag.set()
    
    def _enforce_rate_limits(self, estimated_tokens: int,
                             pbar: tqdm = None,
                             tqdm_lock: threading.Lock = None) -> None:
        '''
        Enforce rate limits for the OpenAI API.
        This method ensures that the number of requests and tokens used
        per minute does not exceed the specified limits.

        :param estimated_tokens: The estimated tokens in the prompt.
        :type estimated_tokens: int

        :param pbar: The tqdm progress bar.
        :type pbar: tqdm

        :param tqdm_lock: The threading lock for the tqdm progress bar.
        :type tqdm_lock: threading.Lock

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

                # update progress bar
                with tqdm_lock:
                    pbar.set_postfix({"Tokens used/60s": tokens_used})

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
                    self.stop_flag.wait(jitter)
                    break

            # wait_time != 0 - enforced rate-limit sleep - when over RPM/TPM
            jittered_wait = wait_time + random.uniform(0.05, 0.25)
            with tqdm_lock:
                pbar.set_description(f"[WAITING] Sleeping {jittered_wait:.2f}s (RPM/TPM limit)")

            self.stop_flag.wait(jittered_wait)
    
    def _process_response(self, response: openai.ChatCompletion,
                          mongo_db_name: str = None,
                          mongo_collection_name: str = None) -> None:
        '''
        Process API response.

        :param response: The response from the OpenAI API.
        :type response: openai.ChatCompletion

        :param mongo_db_name: The MongoDB database name.
        :type mongo_db_name: str

        :param mongo_collection_name: The MongoDB collection name.
        :type mongo_collection_name: str

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
            db_name=mongo_db_name,
            collection_name=mongo_collection_name   
        )

        # parse response
        response_content = ast.literal_eval(response_content)

        # upload response to database
        collection.insert_one(
            response_content
        )
    
    def _call_with_backoff(self, message: list[dict], request_id: int,
                           mongo_db_name: str = None,
                           mongo_collection_name: str = None,
                           pbar: tqdm = None,
                           tqdm_lock: threading.Lock = None,
                           max_retries: int = 5) -> None:
        '''
        Call the OpenAI API with a backoff strategy.

        :param message: The message prompt to be processed.
        :type message: list[dict]

        :param request_id: The request ID.
        :type request_id: int

        :param mongo_db_name: Name of the MongoDB database.
        :type mongo_db_name: str

        :param mongo_collection_name: Name of the MongoDB collection.
        :type mongo_collection_name: str

        :param pbar: The tqdm progress bar.
        :type pbar: tqdm

        :param tqdm_lock: The threading lock for the tqdm progress bar.
        :type tqdm_lock: threading.Lock

        :param max_retries: The maximum number of retries.
        :type max_retries: int

        :return: None
        '''
        retry_count = 0
        while not self.stop_flag.is_set() and retry_count <= max_retries:
            with self.semaphore:
                try:
                    # get prompts
                    system_prompt = message[0]['content']
                    user_prompt = message[1]['content']

                    # estimate tokens
                    prompt = f'{system_prompt}\n{user_prompt}'
                    estimated_tokens = self._estimate_tokens(prompt)

                    # update progress bar
                    with tqdm_lock:
                        pbar.set_description(f"[RUNNING] prompt #{request_id} in thread")
                    
                    # enforce rate limits
                    self._enforce_rate_limits(
                        estimated_tokens,
                        pbar,
                        tqdm_lock
                    )

                    # # make request
                    # response = self.client.chat.completions.create(
                    #     model=self.model_name,
                    #     messages=message
                    # )

                    # self._process_response(response, mongo_db_name, mongo_collection_name)

                    # update token usage log
                    self.token_usage_log.append(
                        (
                            time.time(),
                            estimated_tokens,
                            random.randint(400, 500)
                        )
                    )
                    return
                    
                except RateLimitError:
                    # update progress bar
                    with tqdm_lock:
                        pbar.set_description(f"[RETRY {retry_count}] prompt #{request_id} sleeping")
                    
                    # increment retry count
                    retry_count += 1

                    # calculate sleep time
                    sleep_time = (2 ** retry_count) + random.uniform(0, self.DEFAULT_JITTER)

                    # sleep
                    self.stop_flag.wait(sleep_time)
                
                except Exception as e:
                    # handle unexpected errors
                    with tqdm_lock:
                        pbar.set_description(f"[ERROR] prompt #{request_id} error")
                    
                    # write to log file
                    self._log_write(f"[ERROR] prompt #{request_id} error: {e.__class__.__name__}")

                    return
        
        # exceeded max retries
        if retry_count > max_retries:
            with tqdm_lock:
                pbar.set_description(f"[FAILED] prompt #{request_id} exceeded retries")
            
            # write to log file
            self._log_write(f"[FAILED] prompt #{request_id} exceeded retries")
            return

    def run_parallel_prompt_tasks(self, messages: list,
                                  mongo_db_name: str = None,
                                  mongo_collection_name: str = None) -> None:
        '''
        Run parallel prompt tasks with thread pooling and rate-limiting.

        :param messages: List of message prompts to process.
        :type messages: list

        :param mongo_db_name: Name of the MongoDB database.
        :type mongo_db_name: str

        :param mongo_collection_name: Name of the MongoDB collection.
        :type mongo_collection_name: str

        :return: None
        '''
        signal.signal(signal.SIGINT, self._signal_handler)

        # total tasks
        total_tasks = len(messages)
        with tqdm(total=total_tasks, desc="Processing requests") as pbar:
            with ThreadPoolExecutor(max_workers=30) as executor:
                futures = {
                    executor.submit(
                        self._call_with_backoff,
                        message,
                        i,
                        mongo_db_name,
                        mongo_collection_name,
                        pbar,
                        self.tqdm_lock
                    ): i
                    for i, message in enumerate(messages)
                }
                while futures:
                    done, _ = wait(futures.keys(), timeout=0.2)
                    for future in done:
                        futures.pop(future)
                        try:
                            future.result()
                        except Exception as e:
                            pass
                        pbar.update(1)
    
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
