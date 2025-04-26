# -*- coding: utf-8 -*-

# import modules
import textwrap

# argparse formatter
from argparse import HelpFormatter

class CustomHelpFormatter(HelpFormatter):
    '''
    Custom formatter for argparse help output.
    
    Improves subcommand display and wraps long text for better
    readability.
    '''
    def __init__(self, prog, indent_increment=2, max_help_position=40, width=120):
        super().__init__(prog, indent_increment, max_help_position, width)
    
    def _format_action(self, action):
        if action.choices:
            # get the subaction help strings
            subactions = list(action._get_subactions())

            # format the help for each subcommand
            parts = []
            for subaction in subactions:
                # build the complete help string with proper formatting
                cmd = f"  {subaction.dest:<12} {subaction.help or ''}"
                parts.append(cmd)
            
            # join all parts with newlines
            return "\n".join(parts)
        return super()._format_action(action)

    def _split_lines(self, text, width):
        '''
        Wrap text while preserving manual newlines.
        '''
        if '\n' in text:
            return text.splitlines()
        
        return textwrap.wrap(text, width)
