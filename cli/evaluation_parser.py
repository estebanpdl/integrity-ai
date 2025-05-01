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

    return evaluation_parser
