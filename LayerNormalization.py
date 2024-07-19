import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim = True)
        std = x.std(-1,keepdim=True,unbiased=False)
        return self.gamma * (x-mean) / (std+self.eps) + self.beta
    
features = 512
layer_norm = LayerNorm(features)
input_tensor = torch.rand(1,60,features)
output_tensor = layer_norm(input_tensor)
print(output_tensor.shape)
