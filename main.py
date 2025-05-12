def run_simple_bigram(args):
    from lib import WordList, Encoder, SimpleBigram
    from lib import init_random
    import torch

    init_random(2147483647)

    words = WordList('data/names.txt')
    encoder = Encoder(words.vocabulary)
    model = SimpleBigram(words, encoder)

    def generate_words(model, encoder, count):
        words = []
        for _ in range(count):
            index = encoder.get_index('.')
            chars = []
            while True:
                index = model(index)
                if index == 0:
                    break
                chars.append(encoder.get_char(index))
            words.append(''.join(chars))
        return words

    def calculate_loss(model, words):
        log_sum = 0.0
        count = 0
        for word in words:
            for pair in word.get_pairs():
                log_sum += model.get_log_likelihood(pair)
                count += 1
        return log_sum / count

    def show_stats(model, words):
        for word in words:
            for pair in word.get_pairs():
                count = model.get_count(pair)
                prob = model.get_probability(pair)
                likelihood = model.get_log_likelihood(pair)
                print(f'{pair}: {count=:>8}, {prob=:.4f}, {likelihood=:.4f}')

    print('---')
    show_stats(model, words[:3])
    print('---')
    print(f'{calculate_loss(model, words[:])=:.4f}')
    print('---')
    for word in generate_words(model, encoder, 20):
        print(word)
    print('---')


def run_neuron_bigram(args):
    from lib import WordList, Encoder, NeuronBigram
    from lib import init_random, global_generator
    import torch

    init_random(2147483647)

    words = WordList('data/names.txt')
    encoder = Encoder(words.vocabulary)
    model = NeuronBigram(words.vocabulary_size)

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

    def calculate_loss(labels, predictions):
        probs = predictions[torch.arange(predictions.shape[0]),
                            labels.argmax(1)]
        loss = -probs.log().mean()
        return loss

    def generate_words(model, encoder, count, generator=None):
        if generator is None:
            generator = global_generator

        for index in range(count):
            chars = []
            index = encoder.get_index('.')
            while True:
                probs = model(encoder.get_embedding(index))
                index = torch.multinomial(probs, 1, generator=generator).item()
                if index == 0:
                    break
                chars.append(encoder.get_char(index))
            print(''.join(chars))

    inputs, labels = prepare_data(words.count)

    for iteration in range(500):
        predictions = model(inputs)
        loss = calculate_loss(labels, predictions)
        # print(f'[{iteration:>4}] {loss=:.8f}')

        model.reset_grad()
        loss.backward()
        model.descend(50)

    print(f'{loss=:.8f}')
    print('---')
    generate_words(model, encoder, 20)


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
