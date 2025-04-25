# -*- coding: utf-8 -*-

# import modules
import time
import textwrap

# import argparse
from argparse import (
	ArgumentParser, HelpFormatter, SUPPRESS
)

# Narrative blueprint
from narrative_blueprint import NarrativeBlueprint

# language models
from models import OpenAIGPT

'''
Arguments

'''

class CustomHelpFormatter(HelpFormatter):
    def __init__(self, prog, indent_increment=2, max_help_position=40, width=120):
        super().__init__(prog, indent_increment, max_help_position, width)
    
    def _format_action(self, action):
        # For subparsers only, customize the help output
        if action.choices:
            # Get the subaction help strings
            subactions = list(action._get_subactions())
            # Format the help for each subcommand
            parts = []
            for subaction in subactions:
                # Build the complete help string with proper formatting
                cmd = "  {:<12} {}".format(subaction.dest, subaction.help)
                parts.append(cmd)
            # Join all parts with newlines
            return "\n".join(parts)
        return super()._format_action(action)

    def _split_lines(self, text, width):
        """Wrap text to respect the width but maintain indentation for all lines"""
        if '\n' in text:
            # For text with explicit newlines, respect those
            return text.splitlines()
        return textwrap.wrap(text, width)

parser = ArgumentParser(
    prog='Integrity AI',
    description='Integrity AI: Blueprint & Evaluation Toolkit',
    formatter_class=CustomHelpFormatter,
    add_help=False
)

# help arguments
help_arguments = parser.add_argument_group('Help options')
help_arguments.add_argument(
    '-h',
    '--help',
    action='help',
    default=SUPPRESS,
    help='Show this help message and exit'
)

# creating subparsers
subparsers = parser.add_subparsers(
    title='Available subcommands',
    metavar=''
)

# required subparsers
subparsers.required = True

# blueprint subcommand
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


# evaluation subcommand
evaluation_parser = subparsers.add_parser(
    'evaluation',
    help='Run evaluation',
    formatter_class=CustomHelpFormatter,
    add_help=False
)

# parse arguments
args = vars(parser.parse_args())


# start process
log_text = f'''
> Starting program at: {time.ctime()}

'''
print ('\n\n' + ' '.join(log_text.split()).strip() + '\n\n')

openai_llm = OpenAIGPT(model_name=args['model'])
blueprint = NarrativeBlueprint(
    llm_engine=openai_llm,
    args=args
)

# run blueprint analysis
blueprint.run_blueprint_analysis()


# end process
log_text = f'''
> Ending program at: {time.ctime()}

'''
print ('\n\n' + ' '.join(log_text.split()).strip())
