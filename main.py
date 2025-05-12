def run_simple_bigram(args):
    from lib import WordList, Encoder, SimpleBigram
    from lib import init_random
    import torch

    init_random(2147483647)

    words = WordList('data/names.txt')
    encoder = Encoder(words.vocabulary)
    model = SimpleBigram(words, encoder)

    def prepare_data(count=1):
        input_chars = []
        label_chars = []
        for word in words[:count]:
            for pair in word.get_pairs():
                input_chars.append(pair[0])
                label_chars.append(pair[1])

        inputs = torch.tensor(
            [encoder.get_index(char) for char in input_chars])
        labels = torch.tensor(
            [encoder.get_index(char) for char in label_chars])
        return inputs, labels

    # def show_stats(model, words):
    #     for word in words:
    #         for pair in word.get_pairs():
    #             count = model.get_count(pair)
    #             prob = model.get_probability(pair)
    #             likelihood = model.get_log_likelihood(pair)
    #             print(f'{pair}: {count=:>8}, {prob=:.4f}, {likelihood=:.4f}')

    inputs, labels = prepare_data(words.count)
    predictions, loss = model(inputs, labels)

    print('---')
    print(loss)
    print('---')
    for _ in range(5):
        print(model.generate_word(encoder))


def run_neuron_bigram(args):
    from lib import WordList, Encoder, NeuronBigram
    from lib import init_random, global_generator
    import torch

    init_random(2147483647)

    words = WordList('data/names.txt')
    encoder = Encoder(words.vocabulary)
    neuron_model = NeuronBigram(words.vocabulary_size)

    def prepare_data(count=1):
        input_chars = []
        label_chars = []
        for word in words[:count]:
            for pair in word.get_pairs():
                input_chars.append(pair[0])
                label_chars.append(pair[1])

        inputs = torch.stack(
            [encoder.get_embedding(char) for char in input_chars])
        labels = torch.stack(
            [encoder.get_embedding(char) for char in label_chars])
        return inputs, labels

    inputs, labels = prepare_data(words.count)

    for iteration in range(500):
        predictions, loss = neuron_model(inputs, labels=labels)
        print(f'[{iteration:>4}] {loss=:.8f}')

        neuron_model.reset_grad()
        loss.backward()
        neuron_model.descend(50)

    print('---')
    print(loss)
    print('---')
    for _ in range(5):
        print(neuron_model.generate_word(encoder))



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
