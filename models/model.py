# -*- coding: utf-8 -*-

'''
Defines LanguageModel base class

'''

# search: Sketch of new LanguageModel

# import modules
import os
import threading

# abstract base class
from abc import ABC, abstractmethod

# tqdm bar
from tqdm import tqdm

# LanguageModel abstract base class
class LanguageModel(ABC):
    '''
    LanguageModel abstract base class
    '''
    def __init__(self):
        '''
        Initialize LanguageModel abstract base class
        '''
        # shared threading locks
        self.tqdm_lock = threading.Lock()

        # shared threading event
        self.stop_flag = threading.Event()

    @abstractmethod
    def get_log_file(self) -> str:
        '''
        Get the log file.
        '''
        pass

    def _log_write(self, message: str) -> None:
        '''
        Write a message to the log file.

        :param message: The message to be written to the log file.
        :type message: str

        :return: None
        '''
        # get log file
        log_file = self.get_log_file()

        # create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # write to log file
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    
    def _update_tqdm_description(self, pbar: tqdm, message: str) -> None:
        '''
        Update the tqdm progress bar description safely.

        :param pbar: The tqdm progress bar instance.
        :type pbar: tqdm

        :param message: The message to be displayed in the progress bar.
        :type message: str

        :return: None
        '''
        with self.tqdm_lock:
            pbar.set_description(message)
    
    def _update_tqdm_postfix(self, pbar: tqdm, data: dict) -> None:
        '''
        Update the tqdm progress bar postfix (metrics) safely.

        :param pbar: The tqdm progress bar instance.
        :param data: A dictionary of metric names and values.
        '''
        with self.tqdm_lock:
            pbar.set_postfix(data)
    
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

    @abstractmethod
    def _estimate_tokens(self, prompt: str) -> int:
        '''
        Estimate the number of tokens in the prompt.

        :param prompt: The prompt to be estimated.
        :return: The estimated number of tokens.
        '''
        pass

    @abstractmethod
    def run_parallel_prompt_tasks(self, messages: list) -> None:
        '''
        Process multiple messages in a single call.

        :param messages: List of messages to be processed.
        '''
        pass
