import torch

global global_generator
global_generator = torch.Generator()


def init_random(seed=1337):
    import random

    global global_generator
    global_generator = global_generator.manual_seed(seed)
    random.seed(seed)


def prepare_data(words, transform, input_count=1):
    input_chars = []
    label_chars = []
    for word in words:
        for pair in word.get_pairs(input_count):
            input_chars.append(pair[0])
            label_chars.append(pair[1])

    return transform(input_chars), transform(label_chars)
