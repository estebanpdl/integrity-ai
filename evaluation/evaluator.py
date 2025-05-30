# -*- coding: utf-8 -*-

'''
Evaluation engine: Core engine for evaluating LLM responses to
adversarial prompts.

The EvaluationEngine is responsible for systematically presenting
LLMs with adversarial prompts crafted from real-world disinformation
insights (via Narrative Blueprints).
'''

# import modules
import json
import tomli
import string
import pandas as pd

# import LLMs
from models import *

# MongoDB connection
from databases import MongoDBManager

# LLM Judge
from .llm_judge import LLMJudge

# Evaluation engine class
class EvaluationEngine:
    '''
    Evaluation engine class
    '''
    def __init__(self, args: dict = None):
        '''
        Initialize the Evaluation engine

        :param args: The arguments to be passed to the Evaluation engine
        :type args: dict
        '''
        args = args or {}
        self.args = args

        # MongoDB connection
        self.mongodb_manager = MongoDBManager()
        self.mongo_db_name = self.args.get('mongo_db_name')
        self.mongo_collection_name = self.args.get('mongo_collection_name')

        # load test cases
        claims_path = self.args.get('claims_dataset')
        self.claims_dataset = self._load_claims_dataset(claims_path)

        # check required column in claims dataset
        required_columns = ['claim', 'uuid']
        missing_columns = [
            col for col in required_columns
            if col not in self.claims_dataset.columns
        ]
        if missing_columns:
            raise ValueError(f'Claims dataset missing required column: {missing_columns}')
        
        # LLM Judge
        self.llm_judge = LLMJudge(args=self.args)
    
    def _load_claims_dataset(self, path: str = None) -> pd.DataFrame:
        '''
        Load the claims dataset

        :param path: The path to the claims dataset.
        :type path: str

        :return: A pandas DataFrame containing the claims dataset
        :rtype: pd.DataFrame
        '''
        if path.endswith('.csv'):
            try:
                return pd.read_csv(path, encoding='utf-8')
            except Exception as e:
                raise e
        else:
            try:
                return pd.read_excel(path)
            except Exception as e:
                raise e
    
    def _load_prompts(self, language: str = 'en') -> dict:
        '''
        Load the prompts

        :param language: The language of the prompts
        :type language: str
        
        :return: A dictionary of prompts
        :rtype: dict
        '''
        lang = language.upper()

        # load prompts
        prompts_file = f'./prompts/adversarial_prompts/{lang}.toml'
        with open(prompts_file, 'rb') as file:
            prompts = tomli.load(file)
        
        return prompts
    
    def _build_evaluation_dataset(self) -> tuple:
        '''
        Build the evaluation dataset

        :return: A tuple of related uuids and evaluation dataset
        :rtype: tuple
        '''
        # get prompts
        lang = self.args.get('language', 'en')
        prompts = self._load_prompts(language=lang)

        # build evaluation dataset
        related_uuids = []
        evaluation_dataset = []
        for idx, row in self.claims_dataset.iterrows():
            uuid = row['uuid']
            claim = row['claim']
            audience = row['audience']

            # build test cases
            test_case = {
                'uuid': uuid,
                'claim': claim,
                'audience': audience,
                'control_prompt': {
                    'system': prompts.get(
                        'control_prompt_system', {}
                    ).get('prompt', ''),
                    'message': prompts.get(
                        'control_prompt_message', {}
                    ).get('prompt', '')
                },
                'personas': {}
            }

            # assign the claim to the control prompt message
            test_case['control_prompt']['message'] = string.Template(
                test_case['control_prompt']['message']
            ).substitute(
                viewpoint=claim,
                audience=audience
            )

            # append control prompt to the evaluation dataset
            evaluation_dataset.append(
                [
                    {
                        'role': 'system',
                        'content': test_case['control_prompt']['system']
                    },
                    {
                        'role': 'user',
                        'content': test_case['control_prompt']['message']
                    }
                ]
            )

            # add uuid
            related_uuids.append(uuid)

            for persona, content in prompts.get('personas', {}).items():
                test_case['personas'][persona] = {
                    'system': content.get('system', {}).get('prompt', ''),
                    'message': content.get('message', {}).get('prompt', '')
                }

                # assign the claim to the persona prompt message
                test_case['personas'][persona]['message'] = string.Template(
                    test_case['personas'][persona]['message']
                ).substitute(
                    viewpoint=claim,
                    audience=audience
                )

                # append the persona prompt to the evaluation dataset
                evaluation_dataset.append(
                    [
                        {
                            'role': 'system',
                            'content': test_case['personas'][persona]['system']
                        },
                        {
                            'role': 'user',
                            'content': test_case['personas'][persona]['message']
                        }
                    ]
                )

                # add uuid
                related_uuids.append(uuid)
            
            # upload test case to MongoDB
            self.mongodb_manager.upload_test_case(
                test_case=test_case,
                db_name=self.mongo_db_name,
                collection_name=f'{self.mongo_collection_name}_test_cases_{lang}'
            )
        
        return related_uuids, evaluation_dataset
    
    def _load_evaluation_models(self) -> dict:
        '''
        Load the evaluation models
        '''
        with open('./config/evaluation_models.json', 'r') as file:
            return json.load(file)
    
    def _call_llms(self, uuids: list, evaluation_dataset: list) -> list:
        '''
        Call the LLMs

        :param uuids: List of uuids to process.
        :type uuids: list

        :param evaluation_dataset: Evaluation dataset.
        :type evaluation_dataset: list
        '''
        # get evaluation models
        evaluation_models = self._load_evaluation_models()

        # get imported llms
        llm_instances = {
            'openai': OpenAIGPT,
            'groq': GroqModels
        }
        
        for llm in llm_instances:
            print (f'Calling {llm}...')
            models = evaluation_models[llm]
            for model in models:
                llm_engine = llm_instances[llm](model_name=model)
                llm_engine.run_parallel_prompt_tasks(
                    uuids=uuids,
                    messages=evaluation_dataset,
                    mongo_db_name=self.mongo_db_name,
                    mongo_collection_name=self.mongo_collection_name
                )

            print (f'{llm} complete.')
            print ('')
            
    def run_evaluation(self):
        '''
        Run the evaluation
        '''
        uuids, evaluation_dataset = self._build_evaluation_dataset()

        # call LLMs
        self._call_llms(
            uuids=uuids,
            evaluation_dataset=evaluation_dataset
        )
        
        return
