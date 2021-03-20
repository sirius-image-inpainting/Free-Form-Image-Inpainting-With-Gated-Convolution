import torch
import model.generator


g = model.generator.SNPatchGANGenerator()
t = torch.randn(1, 4, 256, 256)

o = g(t)
print(o.shape)
