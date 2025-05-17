import torch
from .base_encoder import BaseEncoder


class BigramEncoder(BaseEncoder):
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
        super().__init__(vocabulary)
        self._embeddings = torch.eye(len(vocabulary))

    def get_embedding(self, inputs) -> torch.Tensor:
        """ Gets the embedding corresponding to the specified input.

        :param inputs: The input for which the embedding is retrieved. This must
        be a tuple of at least one character. Only the first character is used
        to generate the embedding.
        """

        assert isinstance(inputs, tuple), 'Invalid inputs (arg #1)'
        first = inputs[0]
        return self._embeddings[self.get_index(first)]
