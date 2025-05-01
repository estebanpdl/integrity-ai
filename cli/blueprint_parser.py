# -*- coding: utf-8 -*-

# import argparse
from argparse import (
	ArgumentParser, SUPPRESS
)

# import argparse formatter
from utils.argparse_formatter import CustomHelpFormatter

# create blueprint parser
def create_blueprint_parser(subparsers: ArgumentParser) -> ArgumentParser:
    '''
    Create blueprint parser

    :param subparsers: The subparsers to be used for the blueprint parser.
    :type subparsers: ArgumentParser

    :return: The blueprint parser.
    :rtype: ArgumentParser
    '''
    # blueprint parser
    blueprint_parser = subparsers.add_parser(
        'blueprint',
        help='Generate structured narrative blueprints for disinformation analysis',
        formatter_class=CustomHelpFormatter,
        add_help=False
    )

    # help options
    blueprint_help = blueprint_parser.add_argument_group('Options')
    blueprint_help.add_argument(
        '-h',
        '--help',
        action='help',
        default=SUPPRESS,
        help='Show blueprint command help'
    )

    # blueprint arguments
    blueprint_arguments = blueprint_parser.add_argument_group(
        'Blueprint arguments'
    )

    blueprint_arguments.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        metavar='',
        help='Language model to use for blueprint generation'
    )

    blueprint_arguments.add_argument(
        '--language',
        type=str,
        default='en',
        metavar='',
        help=(
            "Target language for blueprint generation. This guides the model "
            "to create a narrative blueprint in the specified language and "
            "determines which language-specific prompt templates are used. "
            "Available options: 'en' (English), 'es' (Spanish)"
        )
    )

    blueprint_arguments.add_argument(
        '--narrative-path',
        type=str,
        required=True,
        metavar='',
        help=(
            "Path to the narrative CSV file. File must contain the following "
            "columns: `uuid`, `narrative`"
        )
    )

    blueprint_arguments.add_argument(
        '--sample-size',
        type=int,
        metavar='',
        default=None,
        help='Number of narratives to process for blueprint generation'
    )

    # blueprint MongoDB arguments
    blueprint_mongodb_arguments = blueprint_parser.add_argument_group(
        'Blueprint MongoDB arguments'
    )

    blueprint_mongodb_arguments.add_argument(
        '--mongo-db-name',
        type=str,
        required=True,
        default='narrative-blueprint',
        metavar='',
        help='Name of the MongoDB database to store the blueprint'
    )

    blueprint_mongodb_arguments.add_argument(
        '--mongo-collection-name',
        type=str,
        required=True,
        metavar='',
        help='Name of the MongoDB collection to store the blueprint'
    )

    return blueprint_parser
