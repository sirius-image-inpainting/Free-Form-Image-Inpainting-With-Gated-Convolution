import torch
import torchsummary

import model.layers
import model.generator
import model.discriminator


def make_device():
    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device('cpu')


def make_tensor(shape):
    return torch.randn(*shape)


device = make_device()
m = model.discriminator.SNPatchGANDiscriminator()
input_size = (5, 256, 256)


m = m.to(device)
torchsummary.summary(m, input_size)
