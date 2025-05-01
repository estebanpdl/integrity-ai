# -*- coding: utf-8 -*-

# import modules
import time

# import key arguments
from cli import parser

# import runners
from runners import command_dispatcher

def main():
    # parse arguments
    args = vars(parser.parse_args())

    # start process
    log_text = f'''
    > Starting program at: {time.ctime()}

    '''
    print ('\n\n' + ' '.join(log_text.split()).strip() + '\n\n')

    # command dispatcher
    command = args.pop('command')
    command_dispatcher[command](args)

    # end process
    log_text = f'''
    > Ending program at: {time.ctime()}

    '''
    print ('\n\n' + ' '.join(log_text.split()).strip())

if __name__ == '__main__':
    main()
