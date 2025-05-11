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

    def __call__(self, embeddings, generator=None):
        if generator is None:
            generator = global_generator

        logits = self._weights @ embeddings
        sum_index = len(logits.shape) - 1
        print(sum_index)
        probs = torch.exp(logits)
        probs /= probs.sum(sum_index, keepdim=True)
        return probs
