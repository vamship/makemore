import torch
from .word_list import WordList
from .encoder import Encoder


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

    def __call__(self, index, generator=None):
        return torch.multinomial(self._bigram_probs[index],
                                       1,
                                       replacement=True,
                                       generator=generator).item()

    def get_char_pair_count(self, pair):
        row, col = self._get_pair_indices(pair)
        return self._bigram_counts[row, col].item()

    def get_char_pair_prob(self, pair):
        row, col = self._get_pair_indices(pair)
        return self._bigram_probs[row, col].item()

    def show_counts(self, encoder):
        SimpleBigram._show_data(self._bigram_counts, encoder)

    def show_probs(self, encoder):
        SimpleBigram._show_data(self._bigram_probs, encoder)
