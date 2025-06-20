# -*- coding: utf-8 -*-

# embeddings
from embeddings import OpenAIEmbeddingModel, GeminiEmbeddingModel

# import claim analysis
from analysis import ClaimAnalysis

# handle claim analysis
def handle_claim_analysis(args: dict) -> None:
    '''
    Handle the claim analysis command.

    :param args: The arguments to be passed to the claim analysis command.
    :type args: dict

    :return: None
    :rtype: None
    '''
    embedding_model_map = {
        'text-embedding-004': 'Gemini',
        'gemini-embedding-exp-03-07': 'Gemini',
        'text-embedding-ada-002': 'OpenAI',
        'text-embedding-3-small': 'OpenAI',
        'text-embedding-3-large': 'OpenAI'
    }

    # dispatch map for embedding models
    embedding_model_classes = {
        'Gemini': GeminiEmbeddingModel,
        'OpenAI': OpenAIEmbeddingModel,
    }

    # get model name
    model = args['model']
    provider = embedding_model_map[model]

    # select and instantiate embedding model
    try:
        model_class = embedding_model_classes[provider]
        vector_model = model_class(model_name=model)
    except KeyError:
        raise ValueError(
            f'Unsupported embedding model type: {model}'
        )

    # claim analysis
    claim_analysis = ClaimAnalysis(vector_model=vector_model, args=args)

    # run claim analysis
    print(f'Computing embeddings using {provider} {model}...')
    claim_analysis.run_claim_analysis()
