def run_simple_bigram(args):
    from lib import WordList, Encoder, SimpleBigram
    from lib import generate_words, show_stats, calculate_loss
    import torch

    words = WordList('data/names.txt')
    encoder = Encoder(words.vocabulary)
    model = SimpleBigram(words, encoder)

    generator = torch.Generator().manual_seed(1337)

    for word in generate_words(model, encoder, 5, generator):
        print(word)
    print('---')
    show_stats(model, words[:3])
    print('---')
    print(f'{calculate_loss(model, words[:3])=:.4f}')
    print('---')


def run_neuron_bigram(args):
    from lib import WordList, Encoder, NeuronBigram
    from lib import generate_words, show_stats, calculate_loss, init_random
    import torch

    init_random(1337)

    words = WordList('data/names.txt')
    encoder = Encoder(words.vocabulary)
    model = NeuronBigram(words.vocabulary_size)
    xs = torch.stack([encoder.get_embedding(char) for char in words[0].text])
    # print(encoder.get_embedding('.'))
    print(xs)
    print(model(encoder.get_embedding('.')))


if __name__ == '__main__':
    import sys
    import logging

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: '
                        '%(name)10s: '
                        '%(funcName)-15s: '
                        '%(message)s')

    command_map = {
        'simple': (run_simple_bigram, 'Evaluates the simple bigram model'),
        'neuron': (run_neuron_bigram, 'Evaluates the neuron bigram model'),
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
