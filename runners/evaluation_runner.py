# -*- coding: utf-8 -*-

# language models
from models import *

# import evaluation
from evaluation import EvaluationEngine

# handle evaluation
def handle_evaluation(args: dict) -> None:
    '''
    Handle the evaluation command.

    :param args: The arguments to be passed to the evaluation command.
    :type args: dict

    :return: None
    :rtype: None
    '''
    evaluation_engine = EvaluationEngine(args)

    # run evaluation
    print ('Running evaluation...')
    evaluation_engine.run_evaluation()
