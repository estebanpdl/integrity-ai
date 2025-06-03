# -*- coding: utf-8 -*-

# import argparse
from argparse import (
	ArgumentParser, SUPPRESS
)

# import argparse formatter
from utils.argparse_formatter import CustomHelpFormatter

# create evaluation parser
def create_evaluation_parser(subparsers: ArgumentParser) -> ArgumentParser:
    '''
    Create evaluation parser

    :param subparsers: The subparsers to be used for the evaluation parser.
    :type subparsers: ArgumentParser

    :return: The evaluation parser.
    :rtype: ArgumentParser
    '''
    # evaluation parser
    evaluation_parser = subparsers.add_parser(
        'evaluation',
        help='Run evaluation',
        formatter_class=CustomHelpFormatter,
        add_help=False
    )

    # help options
    evaluation_help = evaluation_parser.add_argument_group('Options')
    evaluation_help.add_argument(
        '-h',
        '--help',
        action='help',
        default=SUPPRESS,
        help='Show evaluation command help'
    )

    # evaluation arguments
    evaluation_arguments = evaluation_parser.add_argument_group(
        'Evaluation arguments'
    )

    evaluation_arguments.add_argument(
        '--claims-dataset',
        type=str,
        required=True,
        metavar='',
        help=(
            "The path to the claims dataset. File must contain a column "
            "named `claim`. File must be a CSV or Excel file."
        )
    )

    evaluation_arguments.add_argument(
        '--language',
        type=str,
        choices=['en', 'es'],
        default='en',
        metavar='',
        help=(
            "Target language for evaluation. This guides the model "
            "to evaluate the LLM responses in the specified language and "
            "determines which language-specific prompt templates are used. "
            "Available options: 'en' (English), 'es' (Spanish)"
        )
    )

    evaluation_arguments.add_argument(
        '--test',
        action='store_true',
        help=(
            "Enable test mode. When specified, the evaluation will use "
            "dummy datasets and return dummy responses instead of making "
            "actual API calls. Useful for testing and development."
        )
    )

    # evaluation MongoDB arguments
    evaluation_mongodb_arguments = evaluation_parser.add_argument_group(
        'Evaluation MongoDB arguments'
    )

    evaluation_mongodb_arguments.add_argument(
        '--mongo-db-name',
        type=str,
        required=True,
        metavar='',
        help='The name of the MongoDB database to store LLM responses'
    )

    evaluation_mongodb_arguments.add_argument(
        '--mongo-collection-name',
        type=str,
        required=True,
        metavar='',
        help='The name of the MongoDB collection to store LLM responses'
    )

    # evaluation LLM Judge arguments
    evaluation_llm_judge_arguments = evaluation_parser.add_argument_group(
        'Evaluation LLM Judge arguments'
    )

    evaluation_llm_judge_arguments.add_argument(
        '--judge-model',
        type=str,
        required=True,
        metavar='',
        help='Specifies which language model will serve as the evaluator (LLM judge).'
    )

    return evaluation_parser
