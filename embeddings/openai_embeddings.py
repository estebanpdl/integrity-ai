# -*- coding: utf-8 -*-

# import modules
import time
import tiktoken

# progress bar
from tqdm import tqdm

# dotenv for environment variables
from dotenv import load_dotenv

# import base class
from .base import VectorModel

# OpenAI related modules
from openai import OpenAI, RateLimitError

# MongoDB connection
from databases import MongoDBManager

# Embedding Model class
class OpenAIEmbeddingModel(VectorModel):
    '''
    OpenAIEmbeddingModel class
    '''
    # OpenAI model limits
    MODEL_LIMITS = {
        'text-embedding-ada-002': {
            'rpm': 3000,
            'tpm': 1000000
        },
        'text-embedding-3-small': {
            'rpm': 3000,
            'tpm': 1000000
        },
        'text-embedding-3-large': {
            'rpm': 3000,
            'tpm': 1000000
        }
    }

    # model token encoding
    ENCODING = 'cl100k_base'

    # max tokens per request
    MAX_TOKENS_PER_REQUEST = 600000

    # max number of documents to process in one batch
    DOCS_PER_BATCH = 2000

    def __init__(self, model_name: str):
        '''
        Initialize the OpenAIEmbeddingModel class.

        :param model_name: The name of the OpenAI model to be used.
        :type model_name: str
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
        self.encoding = tiktoken.get_encoding(self.ENCODING)

        # MongoDB connection
        self.mongodb_manager = MongoDBManager()
    
    def estimate_tokens(self, text: str) -> int:
        '''
        Estimate the number of tokens in data.

        :param text: The text to be estimated.
        :type text: str

        :return: The estimated number of tokens.
        :rtype: int
        '''
        # estimate tokens
        return len(self.encoding.encode(text))
    
    def _insert_into_mongodb(self,
                             data: list[dict] = None,
                             mongo_db_name: str = None,
                             mongo_collection_name: str = None) -> None:
        '''
        Insert data into MongoDB.

        :param data: The data to be inserted.
        :type data: list[dict]

        :param mongo_db_name: The name of the MongoDB database.
        :type mongo_db_name: str

        :param mongo_collection_name: The name of the MongoDB collection.
        :type mongo_collection_name: str
        '''
        # insert into MongoDB
        self.mongodb_manager.insert_many(
            db_name=mongo_db_name,
            collection_name=mongo_collection_name,
            data=data
        )
    
    def _safe_embed_request(self,
                            uuids: list = None,
                            batch_data: list[str] = None,
                            mongo_db_name: str = None,
                            mongo_collection_name: str = None,
                            request_id: int = None,
                            retry_max: int = 3) -> list[list[float]]:
        '''
        Safe embedding request.

        :param uuids: The uuids of the documents to be embedded.
        :type uuids: list

        :param batch_data: The batch of text to be embedded.
        :type batch_data: list[str]

        :param mongo_db_name: The name of the MongoDB database.
        :type mongo_db_name: str

        :param mongo_collection_name: The name of the MongoDB collection.
        :type mongo_collection_name: str

        :param request_id: The request ID.
        :type request_id: int

        :param retry_max: The maximum number of retries.
        :type retry_max: int

        :return: The embedded text.
        :rtype: list[list[float]]
        '''
        retry_count = 0
        while retry_count < retry_max:
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_data
                )

                # build data object
                data = [
                    {
                        'uuid': uuids[i],
                        'text': batch_data[i],
                        'embedding': response.data[i].embedding,
                        'model_name': self.model_name,
                        'provider': 'OpenAI'
                    }
                    for i in range(len(uuids))
                ]

                # insert into MongoDB
                self._insert_into_mongodb(
                    data=data,
                    mongo_db_name=mongo_db_name,
                    mongo_collection_name=mongo_collection_name
                )

                return
            except RateLimitError as e:
                retry_count += 1
                wait_time = 60
                time.sleep(wait_time)
        
        # exceeded max retries
        if retry_count >= retry_max:
            self._log_write(f'Failed to compute embeddings for request {request_id}')
            raise RuntimeError('Failed to compute embeddings.')

    def _process_batch(self,
                       uuids: list = None,
                       batch_data: list[str] = None,
                       mongo_db_name: str = None,
                       mongo_collection_name: str = None,
                       rpm_start_time=None,
                       tpm_start_time=None,
                       request_count=0,
                       pbar=None,
                       start_request_id=1,
                       global_token_count: int = 0) -> tuple[float, float, int, int, int]:
        '''
        Process a batch of documents.
        
        :param batch_data: Batch of documents to process
        :type batch_data: list[str]

        :param rpm_start_time: Start time for rate limit tracking
        :type rpm_start_time: float
        
        :param tpm_start_time: Start time for token limit tracking
        :type tpm_start_time: float
        
        :param request_count: Current request count
        :type request_count: int
        
        :param pbar: Progress bar
        :type pbar: tqdm
        
        :param start_request_id: Starting request ID
        :type start_request_id: int
        
        :param global_token_count: Global token count
        :type global_token_count: int

        :return: Tuple of (embeddings, rpm_start_time, tpm_start_time,
            request_count, next_request_id, global_token_count)
        :rtype: tuple[float, float, int, int, int]
        '''
        current_batch = []

        # token count
        token_count = 0

        # request ID
        request_id = start_request_id
        
        # process batch data
        for text in batch_data:
            text_tokens = self.estimate_tokens(text)

            # check token count
            if (
                token_count + text_tokens > self.max_tokens_per_min or
                token_count + text_tokens > self.MAX_TOKENS_PER_REQUEST
            ):
                # compute embeddings
                self._safe_embed_request(
                    uuids=uuids,
                    batch_data=current_batch,
                    mongo_db_name=mongo_db_name,
                    mongo_collection_name=mongo_collection_name,
                    request_id=request_id
                )
                
                # update progress bar for processed batch
                pbar.update(len(current_batch))

                # increment request count
                request_count += 1
                request_id += 1
                current_batch = []

                # check RPM limits
                elapsed_rpm = time.time() - rpm_start_time
                if request_count >= self.max_requests_per_min and elapsed_rpm < 60:
                    pbar.set_description(f'Rate limit pause: {int(60 - elapsed_rpm)}s')
                    time.sleep(60 - elapsed_rpm)
                    rpm_start_time = time.time()
                    request_count = 0
                    pbar.set_description('Computing embeddings')

                # check TPM limits
                elapsed_tpm = time.time() - tpm_start_time
                if elapsed_tpm < 60:
                    pbar.set_description(f'Token limit pause: {int(60 - elapsed_tpm)}s')
                    time.sleep(60 - elapsed_tpm)
                    tpm_start_time = time.time()
                    pbar.set_description('Computing embeddings')

            # add text to batch
            current_batch.append(text)
            token_count += text_tokens
            global_token_count += text_tokens

            # tokens and requests being processed
            pbar.set_postfix(
                {
                    'requests': request_count,
                    'tokens/batch': token_count,
                    'tokens/total': global_token_count
                }
            )

        # compute embeddings for current batch
        if current_batch:
            # compute embeddings
            self._safe_embed_request(
                uuids=uuids,
                batch_data=current_batch,
                mongo_db_name=mongo_db_name,
                mongo_collection_name=mongo_collection_name,
                request_id=request_id
            )

            # update progress bar
            pbar.update(len(current_batch))

            # increment request processes
            request_count += 1
            request_id += 1
        
        return rpm_start_time, tpm_start_time, \
            request_count, request_id, global_token_count
    
    def compute_embeddings(self,
                           uuids: list = None,
                           data: list[str] = None,
                           mongo_db_name: str = None,
                           mongo_collection_name: str = None) -> list[list[float]]:
        '''
        Compute the embeddings for the data.

        :param data: The data to be computed.
        :type data: list[str]

        :return: The computed embeddings.
        :rtype: list[list[float]]
        '''
        # initialize rate limiting trackers
        rpm_start_time = time.time()
        tpm_start_time = time.time()
        request_count = 0
        next_request_id = 1
        global_token_count = 0
        
        # create single progress bar for all batches
        with tqdm(total=len(data), desc="Computing embeddings", unit="text") as pbar:
            # process data in batches
            for batch_start in range(0, len(data), self.DOCS_PER_BATCH):
                batch_end = min(batch_start + self.DOCS_PER_BATCH, len(data))
                batch_data = data[batch_start:batch_end]

                # uuids
                batch_uuids = uuids[batch_start:batch_end]
                
                # process batch
                result = self._process_batch(
                    uuids=batch_uuids,
                    batch_data=batch_data,
                    mongo_db_name=mongo_db_name,
                    mongo_collection_name=mongo_collection_name,
                    rpm_start_time=rpm_start_time,
                    tpm_start_time=tpm_start_time,
                    request_count=request_count,
                    pbar=pbar,
                    start_request_id=next_request_id,
                    global_token_count=global_token_count
                )

                rpm_start_time, tpm_start_time, \
                    request_count, next_request_id, global_token_count = result
        
        return
