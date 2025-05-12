import torch
from .word_list import WordList
from .encoder import Encoder
from .utils import global_generator


class SimpleBigram:

    def __init__(self, word_list, encoder):
        assert isinstance(word_list, WordList), "Invalid word_list (arg #1)"
        assert isinstance(encoder, Encoder), "Invalid encoder (arg #2)"

        self._bigram_counts = torch.zeros(
            (word_list.vocabulary_size, word_list.vocabulary_size),
            dtype=torch.float)
        self._bigram_probs = torch.zeros(
            (word_list.vocabulary_size, word_list.vocabulary_size),
            dtype=torch.float)
        self._encoder = encoder

        for word in word_list:
            for pair in word.get_pairs():
                row, col = self._get_pair_indices(pair)
                self._bigram_counts[row, col] += 1

        sums = self._bigram_counts.sum(1, keepdim=True)
        self._bigram_probs = self._bigram_counts / sums

    def _get_pair_indices(self, pair):
        row, col, *_ = [self._encoder.get_index(char) for char in pair]
        return row, col

    def _show_data(data, encoder):
        row_size, col_size, *_ = data.shape
        for row_index in range(row_size):
            row = data[row_index]
            first = encoder.get_char(row_index)
            labels = []
            probs = []
            for col_index in range(col_size):
                second = encoder.get_char(col_index)
                labels.append(f'{first+second:^8}')
                probs.append(data[row_index, col_index].item())
            print(' '.join(labels))
            print(' '.join([f'{prob:^8.4f}' for prob in probs]))

    def __call__(self, inputs, labels=None, generator=None):
        if generator is None:
            generator = global_generator

        if isinstance(inputs, (int, float)):
            inputs = torch.tensor([inputs], dtype=torch.float)
        elif isinstance(inputs, list):
            inputs = torch.tensor(inputs, dtype=torch.float)
        assert isinstance(inputs, torch.Tensor), 'Invalid inputs (arg #1)'

        predictions = torch.multinomial(self._bigram_probs[inputs.int()],
                                        1,
                                        replacement=True,
                                        generator=generator)
        loss = None
        if labels is not None:
            loss = -torch.log(self._bigram_probs[inputs, labels]).mean()

        return predictions, loss

    def generate_word(self, encoder, generator=None):
        if generator is None:
            generator = global_generator

        chars = []
        index = encoder.get_index('.')
        while True:
            index, _ = self(index)
            index = index.item()
            if index == encoder.get_index('.'):
                break
            chars.append(encoder.get_char(index))
        return ''.join(chars)

    def get_count(self, pair):
        row, col = self._get_pair_indices(pair)
        return self._bigram_counts[row, col].item()

    def get_probability(self, pair):
        row, col = self._get_pair_indices(pair)
        return self._bigram_probs[row, col].item()

    def get_log_likelihood(self, pair):
        row, col = self._get_pair_indices(pair)
        return -torch.log(self._bigram_probs[row, col])

    def show_counts(self, encoder):
        SimpleBigram._show_data(self._bigram_counts, encoder)

    def show_probs(self, encoder):
        SimpleBigram._show_data(self._bigram_probs, encoder)
