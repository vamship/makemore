def run_default(args):
    from lib import WordList, Encoder, SimpleBigram
    import torch

    words = WordList('data/names.txt')
    encoder = Encoder(words.vocabulary)
    model = SimpleBigram(words, encoder)

    generator = torch.Generator().manual_seed(1337)
    for _ in range(5):
        index = encoder.get_index('.')
        chars = []
        while True:
            index = model(index, generator)
            if index == 0:
                break
            chars.append(encoder.get_char(index))

        print(''.join(chars))


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
    command_name = args[1] if len(args) > 1 else ''
    command = command_map.get(command_name, None)
    if command is not None:
        command[0](args[1:])
    else:
        print(f'Invalid args: {sys.argv[1:]}\n')
        print(f'Usage: {args[0]} <command_name> [command_args]')
        print('Supported commands:')
        for key in command_map:
            command, desc = command_map[key]
            print(f' {key:<10}: {desc}')
        print('')
        sys.exit(1)
