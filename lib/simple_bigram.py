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

        for word in word_list:
            for pair in word.get_pairs(1):
                row, col = self._get_pair_indices(pair, encoder)
                self._bigram_counts[row, col] += 1

        sums = self._bigram_counts.sum(1, keepdim=True)
        self._bigram_probs = self._bigram_counts / sums

    def _get_pair_indices(self, pair, encoder):
        row = encoder.get_index(pair[0][0])
        col = encoder.get_index(pair[1])
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
            inputs = torch.tensor([[inputs]], dtype=torch.float)
        elif isinstance(inputs, list):
            inputs = torch.tensor([inputs], dtype=torch.float)
        assert isinstance(inputs, torch.Tensor), 'Invalid inputs (arg #1)'

        # Use -1 because we're expecting a single input
        indices = inputs[:, -1].int()
        predictions = torch.multinomial(self._bigram_probs[indices],
                                        1,
                                        replacement=True,
                                        generator=generator)
        loss = None
        if labels is not None:
            loss = -torch.log(self._bigram_probs[indices, labels]).mean()

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

    def get_count(self, pair, encoder):
        row, col = self._get_pair_indices(pair, encoder)
        return self._bigram_counts[row, col].item()

    def get_probability(self, pair, encoder):
        row, col = self._get_pair_indices(pair, encoder)
        return self._bigram_probs[row, col].item()

    def get_log_likelihood(self, pair, encoder):
        row, col = self._get_pair_indices(pair, encoder)
        return -torch.log(self._bigram_probs[row, col])

    def show_counts(self, encoder):
        SimpleBigram._show_data(self._bigram_counts, encoder)

    def show_probs(self, encoder):
        SimpleBigram._show_data(self._bigram_probs, encoder)
