# -*- coding: utf-8 -*-

# language models
from models import OpenAIGPT

# import blueprint
from narrative_blueprint import NarrativeBlueprint

# handle blueprint
def handle_blueprint(args: dict) -> None:
    '''
    Handle the blueprint command.
    '''
    model = args['model']

    # llm engine
    llm_engine = OpenAIGPT(model_name=model)

    # blueprint
    blueprint = NarrativeBlueprint(
        llm_engine=llm_engine,
        args=args
    )

    # run blueprint
    blueprint.run_blueprint_analysis()
