import torch
import torchsummary
import model.discriminator as d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = d.SNPatchGANDiscriminator().to(device)
input_size = (5, 256, 256)

torchsummary.summary(model, input_size)
