def run_simple_bigram(args):
    from lib import WordList, BigramEncoder, SimpleBigram
    from lib import init_random, prepare_data
    import torch

    init_random(2147483647)

    words = WordList('data/names.txt')
    encoder = BigramEncoder(words.vocabulary)
    model = SimpleBigram(words, encoder)
    # def show_stats(model, words):
    #     for word in words:
    #         for pair in word.get_pairs():
    #             count = model.get_count(pair)
    #             prob = model.get_probability(pair)
    #             likelihood = model.get_log_likelihood(pair)
    #             print(f'{pair}: {count=:>8}, {prob=:.4f}, {likelihood=:.4f}')

    transform = lambda chars: torch.tensor(
        [encoder.get_index(char) for char in chars])

    inputs, labels = prepare_data(words[:], transform)
    predictions, loss = model(inputs, labels)

    print('=== Simple model ===')
    print(loss.item())
    print('--- Words ---')
    for _ in range(5):
        print(model.generate_word(encoder))


def run_neuron_bigram(args):
    from lib import WordList, BigramEncoder, NeuronBigram, SimpleBigram
    from lib import init_random, prepare_data
    import torch

    init_random(2147483647)

    words = WordList('data/names.txt')
    encoder = BigramEncoder(words.vocabulary)
    neuron_model = NeuronBigram(words.vocabulary_size)

    transform = lambda chars: torch.stack(
        [encoder.get_embedding(char) for char in chars])
    inputs, labels = prepare_data(words[:], transform)

    loss = None
    for iteration in range(50):
        predictions, loss = neuron_model(inputs, labels=labels)
        print(f'[{iteration:>4}] {loss=:.8f}')

        neuron_model.reset_grad()
        loss.backward()
        neuron_model.update(50)

    print('=== Neuron model ===')
    print(f'loss={loss.item() if loss is not None else "None"}')
    print('--- Words ---')
    init_random(2147483647)
    for _ in range(5):
        print(neuron_model.generate_word(encoder))


def run_sandbox(args):
    from lib import WordList, BigramEncoder, SimpleBigram, NeuronBigram
    from lib import init_random, prepare_data
    import torch

    init_random(2147483647)

    words = WordList('data/names.txt')
    encoder = BigramEncoder(words.vocabulary)
    neuron_model = NeuronBigram(words.vocabulary_size)
    simple_model = SimpleBigram(words, encoder)

    transform = lambda chars: torch.stack(
        [encoder.get_embedding(char) for char in chars])
    inputs, labels = prepare_data(words[:], transform)

    loss = None
    for iteration in range(10):
        predictions, loss = neuron_model(inputs, labels=labels)
        print(f'[{iteration:>4}] {loss=:.8f}')

        neuron_model.reset_grad()
        loss.backward()
        neuron_model.update(50)

    print('=== Neuron model ===')
    print(f'loss={loss.item() if loss is not None else "None"}')
    print('--- Words ---')
    init_random(2147483647)
    for _ in range(5):
        print(neuron_model.generate_word(encoder))

    print('=== Simple model ===')
    print('--- Words ---')
    init_random(2147483647)
    for _ in range(5):
        print(simple_model.generate_word(encoder))


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
