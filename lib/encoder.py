import torch


class Encoder:

    def __init__(self, vocabulary):
        assert isinstance(vocabulary,
                          (str, list)), 'Invalid vocabulary (arg #1)'
        self._char_lookup = list(vocabulary)
        self._embeddings = torch.eye(len(vocabulary))
        self._char_map = {}
        for index, char in enumerate(self._char_lookup):
            self._char_map[char] = index

    def __repr__(self):
        return f'Encoder({len(self._char_lookup)})'

    def get_index(self, char):
        if isinstance(char, str):
            return self._char_map.get(char, None)
        elif isinstance(char, tuple):
            return self.get_index(char[0])  # Assumes a bigram
        elif isinstance(char, list):
            return [self.get_index(ch) for ch in char]
        assert False, 'Invalid char (arg #1)'

    def get_char(self, index):
        assert index >= 0, 'Invalid index (arg #1)'
        return self._char_lookup[index]

    def get_embedding(self, index):
        if isinstance(index, (int, float)):
            return self._embeddings[index]
        if isinstance(index, str):
            return self.get_embedding(self.get_index(index))
        elif isinstance(index, tuple):
            return self.get_embedding(index[0])  # Assumes a bigram
        elif isinstance(index, list):
            return [self.get_embedding(i) for i in index]
        assert index >= 0, 'Invalid index (arg #1)'

    def get_char_from_embedding(self, embedding):
        assert isinstance(embedding,
                          torch.Tensor), 'Invalid embedding (arg #1)'
        assert embedding.shape == self._embeddings.shape[1:], \
            'Invalid embedding shape (arg #1)'
        index = torch.argmax(embedding)
        return self.get_char(index)
