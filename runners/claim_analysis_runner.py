# -*- coding: utf-8 -*-

# import modules
import pandas as pd

# embeddings
from embeddings import OpenAIEmbeddingModel

# import claim analysis
from analysis import ClaimAnalysis

# handle claim analysis
def handle_claim_analysis(args: dict) -> None:
    '''
    Handle the claim analysis command.
    '''
    model = args['model']
    vector_model = OpenAIEmbeddingModel(model_name=model)

    # claim analysis
    claim_analysis = ClaimAnalysis(vector_model=vector_model, args=args)

    # run claim analysis
    claim_analysis.run_claim_analysis()
