def run_default(args):
    pass


if __name__ == '__main__':
    import sys
    import logging

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: '
                        '%(name)10s: '
                        '%(funcName)-15s: '
                        '%(message)s')

    command_map = {
        'default': (run_default, 'Runs the default command'),
    }
    args = sys.argv
    command_name = args[0] if len(args) > 1 else ''
    command = command_map.get(command_name, None)
    if command is not None:
        command(args[1:])
    else:
        print(f'Usage: {args[0]} <command_name> [command_args]')
        print('Supported commands:')
        for key in command_map:
            command, desc = command_map[key]
            print(f' {key:<10}: {desc}')
        print('')
        sys.exit(1)
