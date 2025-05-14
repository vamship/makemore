import torch


class BigramEncoder:
    """
    Encoder for bigrams. Provides methods to convert characters to indices
    within a vocabulary, and to retrieve embeddings for those indices.
    """

    def __init__(self, vocabulary):
        """
        Initializes the BigramEncoder with a given vocabulary.

        @param vocabulary: A string or list of characters representing the
        vocabulary that the encoder operates over.
        """
        assert isinstance(vocabulary,
                          (str, list)), 'Invalid vocabulary (arg #1)'
        self._char_lookup = list(vocabulary)
        self._embeddings = torch.eye(len(vocabulary))
        self._char_map = {}
        for index, char in enumerate(self._char_lookup):
            self._char_map[char] = index

    def __repr__(self) -> str:
        """ Representation of the encoder. """
        return f'Encoder({len(self._char_lookup)})'

    def get_index(self, char):
        """
        Gets the index of a character in the vocabulary.

        @param char: The input for whcihc the appropriate index will be
        returned. This can be a single character, or a tuple or a list of
        tuples/strings. If a tuple is provided, the first element of the tuple
        is used to generate the index. If a list is provided, every element of
        the list is used to generate an array of indices.

        @return: The index of the character in the vocabulary.
        """
        if isinstance(char, str):
            return self._char_map[char]
        elif isinstance(char, tuple):
            return self.get_index(char[0])  # Assumes a bigram
        elif isinstance(char, list):
            return [self.get_index(ch) for ch in char]

        assert False, 'Invalid char (arg #1)'

    def get_char(self, index):
        """
        Gets the character corresponding to the specified vocabulary index.

        @param index: The index of the character in the vocabulary.
        """
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
        """
        Gets the character corresponding to the specified embedding.

        @param embedding: The embedding of the character in the vocabulary. The
        embedding must be a one-dimensional tensor with a size equal to the
        vocabulary length.
        """
        assert isinstance(embedding,
                          torch.Tensor), 'Invalid embedding (arg #1)'
        assert embedding.shape == self._embeddings.shape[1:], \
            'Invalid embedding shape (arg #1)'

        index = torch.argmax(embedding)
        return self.get_char(index)
