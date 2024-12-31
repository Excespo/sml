import torch

from sml.model import FeedForwardNetwork

model = FeedForwardNetwork()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(model)

input = torch.randn(100, 7).to(device)
output = model(input)

print(output.shape)