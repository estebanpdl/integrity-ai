# -*- coding: utf-8 -*-

# import modules
import time
import random

# OpenAI
from openai import RateLimitError

# MongoDB connection
from databases import MongoDBManager

# Narrative blueprint
from narrative_blueprint import NarrativeBlueprint

# OpenAI class
from models import OpenAIGPT

# # MongoDBManager class
# mongodb_manager = MongoDBManager()
# collection = mongodb_manager.get_collection(
#     'narrative-blueprint',
#     'gpt-4o-mini'
# )


# sample_document = { "uuid": "f0aabce3-5e06-4754-a66a-a4a9c69b75bb", "type": "sample" }

# # insert sample document
# collection.insert_one(sample_document)

# uuids = mongodb_manager.get_collected_uuids(
#     'narrative-blueprint',
#     'gpt-4o-mini'
# )

openai_llm = OpenAIGPT(model_name='gpt-4o-mini')
blueprint = NarrativeBlueprint(
    llm_engine=openai_llm,
    args={'narrative_path': '../llm-evaluations/data/eval_narratives.csv'}
)

response = blueprint.run_blueprint_analysis()
print (response)
