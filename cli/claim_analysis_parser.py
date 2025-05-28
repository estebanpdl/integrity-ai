# -*- coding: utf-8 -*-

# import argparse
from argparse import (
	ArgumentParser, SUPPRESS
)

# import argparse formatter
from utils.argparse_formatter import CustomHelpFormatter

# create claim analysis parser
def create_claim_analysis_parser(subparsers: ArgumentParser) -> ArgumentParser:
    '''
    Create a claim analysis parser

    :param subparsers: The subparsers to be used for the claim analysis parser.
    :type subparsers: ArgumentParser

    :return: The claim analysis parser.
    :rtype: ArgumentParser
    '''
    # claim analysis parser
    claim_analysis_parser = subparsers.add_parser(
        'claim-analysis',
        help='Analyze extracted claims using embeddings and community detection',
        formatter_class=CustomHelpFormatter,
        add_help=False
    )

    # help options
    claim_analysis_help = claim_analysis_parser.add_argument_group('Options')
    claim_analysis_help.add_argument(
        '-h',
        '--help',
        action='help',
        default=SUPPRESS,
        help='Show claim-analysis command help'
    )

    # claim-analysis arguments
    claim_analysis_arguments = claim_analysis_parser.add_argument_group(
        'Claim analysis arguments'
    )

    claim_analysis_arguments.add_argument(
        '--model',
        type=str,
        choices=[
            'text-embedding-3-small', 'text-embedding-3-large',
            'text-embedding-ada-002', 'text-embedding-004'
        ],
        default='text-embedding-3-small',
        metavar='',
        help=(
            "Embedding model to use for claim analysis. Available options: "
            "OpenAI 'text-embedding-3-small', 'text-embedding-3-large', "
            "'text-embedding-ada-002', and Gemini 'text-embedding-004'"
        )
    )

    claim_analysis_arguments.add_argument(
        '--claims-path',
        type=str,
        required=True,
        metavar='',
        help='Path to the claims CSV file.'
    )

    claim_analysis_arguments.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        metavar='',
        help='Path where the claim analysis results will be saved'
    )

    claim_analysis_arguments.add_argument(
        '--threshold',
        type=float,
        default=0.75,
        metavar='',
        help='Threshold for the similarity analysis'
    )

    # claim analysis MongoDB arguments
    claim_analysis_mongodb_arguments = claim_analysis_parser.add_argument_group(
        'Claim analysis MongoDB arguments'
    )

    claim_analysis_mongodb_arguments.add_argument(
        '--mongo-db-name',
        type=str,
        required=True,
        default='claim-analysis',
        metavar='',
        help='Name of the MongoDB database to store the claim analysis'
    )

    claim_analysis_mongodb_arguments.add_argument(
        '--mongo-collection-name',
        type=str,
        required=True,
        default='embeddings',
        metavar='',
        help='Name of the MongoDB collection to store the claim analysis'
    )

    return claim_analysis_parser
