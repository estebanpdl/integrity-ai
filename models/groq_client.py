# -*- coding: utf-8 -*-

# import modules
import os
import time
import signal
import random
import traceback

# dotenv for environment variables
from dotenv import load_dotenv

# multithreading
from concurrent.futures import ThreadPoolExecutor, wait

# tqdm bar
from tqdm import tqdm

# import base class
from .model import LanguageModel

# Groq module
from groq import Groq
from groq.types.chat.chat_completion import ChatCompletion

# MongoDB connection
from databases import MongoDBManager

# Groq class
class GroqModels(LanguageModel):
    '''
    GroqModels class
    '''
    def __init__(self, model_name: str):
        '''
        Initialize the GroqModels class.

        :param model_name: The name of the model to be used.
        '''
        super().__init__(provider='groq', model_name=model_name)
        
        # load environment variables
        env_file_path = './config/.env'
        load_dotenv(env_file_path)

        # Groq client
        self.client = Groq(
            api_key=os.getenv('GROQ_API_KEY')
        )

        # Groq model
        self.model_name = model_name

        # log file
        self.log_file = './logs/groq_client.log'

        # MongoDB connection
        self.mongodb_manager = MongoDBManager()
    
    def get_log_file(self) -> str:
        '''
        Get the log file.

        :return: The log file.
        :rtype: str
        '''
        return self.log_file

    def _estimate_tokens(self, prompt: str) -> int:
        '''
        Estimate the number of tokens in a prompt.

        :param prompt: The prompt to be estimated.
        :type prompt: str
        '''
        return len(prompt) // 2
    
    def _process_response(self,
                          uuid: str,
                          system_prompt: str,
                          user_prompt: str,
                          response: ChatCompletion,
                          mongo_db_name: str = None,
                          mongo_collection_name: str = None) -> None:
        '''
        Process API response.

        :param uuid: The uuid of the narrative that has been processed.
        :type uuid: str

        :param system_prompt: The system prompt used to generate the response.
        :type system_prompt: str

        :param user_prompt: The user prompt used to generate the response.
        :type user_prompt: str

        :param response: The response from the Groq API.
        :type response: ChatCompletion

        :param mongo_db_name: Name of the MongoDB database.
        :type mongo_db_name: str

        :param mongo_collection_name: Name of the MongoDB collection.
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

        # build response content data
        response_content = {
            'uuid': uuid,
            'model': self.model_name,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'response': response_content,
            'prompt_tokens_used': prompt_tokens_used,
            'completion_tokens_used': completion_tokens_used,
            'total_tokens_used': prompt_tokens_used + completion_tokens_used
        }

        # get collection
        collection = self.mongodb_manager.get_collection(
            db_name=mongo_db_name,
            collection_name=mongo_collection_name
        )

        # upload response to database
        collection.insert_one(response_content)
    
    def _call_with_backoff(self, request_id: int,
                           uuid: str,
                           message: list[dict],
                           mongo_db_name: str = None,
                           mongo_collection_name: str = None,
                           pbar: tqdm = None,
                           max_retries: int = 5) -> None:
        '''
        Call the Groq API with a backoff strategy.

        :param request_id: The request ID.
        :type request_id: int

        :param uuid: The uuid of the narrative that is being processed.
        :type uuid: str

        :param message: The message prompt to be processed.
        :type message: list[dict]

        :param mongo_db_name: Name of the MongoDB database.
        :type mongo_db_name: str

        :param mongo_collection_name: Name of the MongoDB collection.
        :type mongo_collection_name: str

        :param pbar: The tqdm progress bar.
        :type pbar: tqdm

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
                    self._update_tqdm_description(
                        pbar,
                        f'[RUNNING] prompt #{request_id} in thread'
                    )

                    # enforce rate limits
                    self._enforce_rate_limits(
                        estimated_tokens,
                        pbar
                    )

                    # make request
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=message,
                        temperature=self.TEMPERATURE
                    )

                    # process response
                    self._process_response(
                        uuid,
                        system_prompt,
                        user_prompt,
                        response,
                        mongo_db_name,
                        mongo_collection_name
                    )

                    return
                        
                except Exception as e:
                    # handle unexpected errors
                    self._update_tqdm_description(
                        pbar,
                        f'[ERROR] prompt #{request_id} error'
                    )

                    # capture full traceback
                    tb_str = traceback.format_exc()
                    
                    # write to log file
                    e_name = e.__class__.__name__
                    self._log_write(
                        f'[ERROR] prompt #{uuid} error: {e_name} - {e}\nTraceback:\n{tb_str}'
                    )

                    # increment retry count
                    retry_count += 1
                    random_jitter = random.uniform(0, self.DEFAULT_JITTER)
                    sleep_time = (2 ** retry_count) + random_jitter
                    self.stop_flag.wait(sleep_time)
        
        # exceeded max retries
        if retry_count > max_retries:
            self._update_tqdm_description(
                pbar,
                f'[FAILED] prompt #{request_id} exceeded retries'
            )

            # write to log file
            self._log_write(f"[FAILED] prompt #{uuid} exceeded retries.")
            return

    def run_parallel_prompt_tasks(self,
                                  uuids: list = None,
                                  messages: list = None,
                                  mongo_db_name: str = None,
                                  mongo_collection_name: str = None) -> None:
        '''
        Run parallel prompt tasks with thread pooling and rate-limiting.

        :param uuids: List of uuids to process.
        :type uuids: list

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
                        i,
                        uuid,
                        message,
                        mongo_db_name,
                        mongo_collection_name,
                        pbar
                    ): i
                    for i, (uuid, message) in enumerate(zip(uuids, messages))
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

