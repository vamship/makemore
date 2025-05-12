import torch
from .utils import global_generator


class NeuronBigram:

    def __init__(self, vocabulary_size, generator=None):
        if generator is None:
            generator = global_generator

        self._weights = torch.randn((vocabulary_size, vocabulary_size),
                                    dtype=torch.float,
                                    requires_grad=True,
                                    generator=generator)

    def __call__(self, inputs, labels=None, generator=None):
        if generator is None:
            generator = global_generator

        logits = inputs @ self._weights
        sum_index = len(logits.shape) - 1
        counts = torch.exp(logits)
        probs = counts / counts.sum(sum_index, keepdim=True)
        loss = None
        if labels is not None:
            loss = probs[torch.arange(probs.shape[0]), labels.argmax(1)]
            loss = -loss.log().mean()
        return probs, loss

    def generate_word(self, encoder, generator=None):
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
        self._weights.grad = None

    def descend(self, delta):
        self._weights.data += -delta * self._weights.grad
