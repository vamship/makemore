from typing import Callable
import torch
import random

global global_generator
global_generator = torch.Generator()


def init_random(seed: int = 1337):
    """ Initializes a global random number generator with a seed.

    This method can be used to set random seed values to ensure reproducibility
    in experiments. It sets both the PyTorch global generator and the Python
    random module's seed.

    :param seed: The seed value to initialize the random number generator.
    """

    global global_generator
    global_generator = global_generator.manual_seed(seed)
    random.seed(seed)


def prepare_data(words,
                 transform: Callable[[list, list], tuple[torch.Tensor]],
                 input_count: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """ Prepares a dataset of words for training.

    This function takes a list of words and transforms them into a set of pairs
    of inputs and expected outputs (labels). The number of characters in the
    input can be specified using the input_count parameter.

    Outputs can be transformed using a provided transform function, which is
    applied to both the input and label characters.

    :param words: A list of Word objects representing the words to be used to
    generate the dataset.
    :param transform: A function that transforms the input and label characters
    into a format suitable for training.
    :param input_count: The number of characters to be used to generate the
    input set.
    """

    input_chars = []
    label_chars = []
    for word in words:
        for pair in word.get_pairs(input_count):
            input_chars.append(pair[0])
            label_chars.append(pair[1])

    return transform(input_chars), transform(label_chars)
