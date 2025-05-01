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
        default='text-embedding-3-large',
        metavar='',
        help='Embedding model to use for claim analysis'
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

    return claim_analysis_parser
