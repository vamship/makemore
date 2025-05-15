import torch
from .utils import global_generator, prepare_data
from .bigram_encoder import BigramEncoder
from .word_list import WordList


class NeuronBigram:
    """ Very basic neuron based bigram model based on a single layer of neurons.

    This model is a simple feedforward neural network with a single layer and no
    nonlinearites.
    """

    def __init__(self,
                 vocabulary_size: int,
                 generator: torch.Generator = None):
        """ Initializes the model with random weights.
        Weight generation will use the given generator if provided, or the
        global generator if not.

        :param vocabulary_size: The size of the vocabulary.
        :param generator: A torch generator object used to generate random
        weights.
        """

        assert isinstance(vocabulary_size,
                          int), "Invalid vocabulary_size (arg #1)"
        if generator is None:
            generator = global_generator

        self._weights = torch.randn((vocabulary_size, vocabulary_size),
                                    dtype=torch.float,
                                    requires_grad=True,
                                    generator=generator)

    def __repr__(self) -> str:
        """ Representation of the model. """
        return f'NeuronBigram({self._weights.shape})'

    def __call__(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """ Evaluates the model on the given inputs and provides a probability
        of the expected outcome.

        If labels are provided, loss is calculated using average negative log
        loss.

        :param inputs: Inputs to the model, specified as a tensor of embeddings.
        :param labels: Labels to use for loss calculation. If None, no loss is
        returned
        """

        assert isinstance(inputs, torch.Tensor), 'Invalid inputs (arg #1)'
        assert inputs.shape[-1] == self._weights.shape[
            0], 'Input has an invalid shape (arg #1)'
        if labels is not None:
            assert isinstance(labels, torch.Tensor), 'Invalid labels (arg #2)'
            assert inputs.shape[0] == labels.shape[
                0], 'Input and label shapes do not match (arg #1, #2)'

        # logits = self._weights[torch.argmax(inputs, len(inputs.shape) - 1)]
        logits = inputs @ self._weights
        sum_index = len(logits.shape) - 1
        counts = torch.exp(logits)
        probs = counts / counts.sum(sum_index, keepdim=True)
        loss = None
        if labels is not None:
            loss = probs[torch.arange(probs.shape[0]), labels.argmax(1)]
            loss = -loss.log().mean()
        return probs, loss

    def generate_word(self,
                      encoder: BigramEncoder,
                      generator: torch.Generator = None) -> str:
        """ Generates a word using the model.

        :param encoder: The encoder used to convert indices to characters.
        :param generator: A torch generator object used to randomly select the
        next character based on model outputs.

        :return: The generated word as a string.
        """

        assert isinstance(encoder, BigramEncoder), "Invalid encoder (arg #1)"

        if generator is None:
            generator = global_generator

        chars = []
        index = encoder.get_index('.')
        while True:
            probs, _ = self(encoder.get_embedding(index))
            index = torch.multinomial(probs,
                                      1,
                                      replacement=True,
                                      generator=generator).item()
            if index == encoder.get_index('.'):
                break
            chars.append(encoder.get_char(index))
        return ''.join(chars)

    def reset_grad(self):
        """ Resets the gradient of the network.
        This is necessary to avoid accumulating gradients from multiple gradient
        calculations.
        """

        self._weights.grad = None

    def update(self, delta: float):
        """ Updates the weights of the network in the direction of the negative
        gradient, using the specified step size

        :param delta: The step size to use for the update.
        """

        self._weights.data += -delta * self._weights.grad

    def prepare_data(self, words: WordList,
                     encoder) -> tuple[torch.Tensor, torch.Tensor]:
        """ Prepares data that can be used to train this model.

        :param words: A WordList object containing the words to be used to
        generate the dataset.
        :param encoder: The encoder used to convert characters to embeddings.
        :return: A tuple containing the input and label tensors.
        """

        transform = lambda chars: torch.stack(
            [encoder.get_embedding(char) for char in chars])
        return prepare_data(words[:], transform)
