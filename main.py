def run_simple_bigram(args, words = None, encoder = None):
    from lib import WordList, BigramEncoder, SimpleBigram
    from lib import init_random
    import torch

    init_random(2147483647)

    if words is None:
        words = WordList('data/names.txt')

    if encoder is None:
        encoder = BigramEncoder(words.vocabulary)

    simple_model = SimpleBigram(words, encoder)

    inputs, labels = simple_model.prepare_data(words[:], encoder)
    predictions, loss = simple_model(inputs, labels)

    print('=== Simple model ===')
    print(loss.item())

    return simple_model


def run_neuron_bigram(args, words = None, encoder = None):
    from lib import WordList, BigramEncoder, NeuronBigram, SimpleBigram
    from lib import init_random
    import torch

    init_random(2147483647)

    if words is None:
        words = WordList('data/names.txt')

    if encoder is None:
        encoder = BigramEncoder(words.vocabulary)

    neuron_model = NeuronBigram(words.vocabulary_size)

    inputs, labels = neuron_model.prepare_data(words[:], encoder)
    loss = None
    for iteration in range(500):
        predictions, loss = neuron_model(inputs, labels=labels)
        # print(f'[{iteration:>4}] {loss=:.8f}')

        neuron_model.reset_grad()
        loss.backward()
        neuron_model.update(50)

    print('=== Neuron model ===')
    print(f'loss={loss.item() if loss is not None else "None"}')
    return neuron_model


def run_sandbox(args):
    from lib import init_random, WordList, BigramEncoder

    words = WordList('data/names.txt')
    encoder = BigramEncoder(words.vocabulary)
    neuron_model = run_neuron_bigram(args, words, encoder)
    simple_model = run_simple_bigram(args, words, encoder)

    for model in [simple_model, neuron_model]:
        init_random(2147483647)
        print(f'--- Words {model}---')
        for _ in range(5):
            print(model.generate_word(encoder))


if __name__ == '__main__':
    import sys
    import logging

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: '
                        '%(name)10s: '
                        '%(funcName)-15s: '
                        '%(message)s')

    command_map = {
        'sandbox': (run_sandbox, 'Runs sandbox code used to evaluate ideas'),
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
