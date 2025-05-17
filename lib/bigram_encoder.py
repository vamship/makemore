import torch


class BigramEncoder:
    """
    Encoder for bigrams. Provides methods to convert characters to indices
    within a vocabulary, and to retrieve embeddings for those indices.
    """

    def __init__(self, vocabulary: str | list[str]):
        """
        Initializes the BigramEncoder with a given vocabulary.

        :param vocabulary: A string or list of characters representing the
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

    def get_index(self, char: str) -> int:
        """ Gets the index of a character in the vocabulary.

        :param char: The input for whcihc the appropriate index will be
        returned. This can be a single character, or a tuple or a list of
        tuples/strings. If a tuple is provided, the first element of the tuple
        is used to generate the index. If a list is provided, every element of
        the list is used to generate an array of indices.

        :return: The index of the character in the vocabulary.
        """

        assert isinstance(char, str), 'Invalid char (arg #1)'
        return self._char_map[char]

    def get_char(self, index: int) -> str:
        """ Gets the character corresponding to the specified vocabulary index.

        :param index: The index of the character in the vocabulary.
        """

        assert index >= 0, 'Invalid index (arg #1)'
        return self._char_lookup[index]

    def get_embedding(self, inputs) -> torch.Tensor:
        """ Gets the embedding corresponding to the specified input.

        :param inputs: The input for which the embedding is retrieved. This must
        be a tuple of at least one character. Only the first character is used
        to generate the embedding.
        """

        assert isinstance(inputs, tuple), 'Invalid inputs (arg #1)'
        first = inputs[0]
        return self._embeddings[self.get_index(first)]
