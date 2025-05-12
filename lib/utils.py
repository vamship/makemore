import torch

global global_generator
global_generator = torch.Generator()

def init_random(seed=1337):
    import random

    global global_generator
    global_generator = global_generator.manual_seed(seed)
    random.seed(seed)
